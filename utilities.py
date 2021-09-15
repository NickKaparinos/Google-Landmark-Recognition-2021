"""
Nick Kaparinos
Google Landmark Recognition 2021
Kaggle Competition
"""
import numpy as np
import pandas as pd
from random import shuffle, seed, sample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import os
import time
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from joblib import Parallel, delayed
from operator import itemgetter
from torch.utils.data import DataLoader
from arc_face import *


class PytorchTransferModel(nn.Module):
    def __init__(self, input_channels=3, print_shape=False, n_classes=81313):
        super().__init__()
        # Use efficientnet
        self.model = EfficientNet.from_pretrained('efficientnet-b1')
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(11520, 512)
        self.fc2 = nn.Linear(512, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.arc_face = ArcMarginProduct(512, n_classes, s=30, m=0.5)

    def forward(self, x):
        x, y = x
        x = self.model.extract_features(x.float())
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.Flatten()(x)
        x = self.batch_norm(self.fc1(x))
        x = F.relu(x)
        x = self.arc_face(x, y)
        return x

    def extract_features(self, x):
        x = self.model.extract_features(x.float())
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.Flatten()(x)
        x = self.batch_norm(self.fc1(x))
        x = F.relu(x)
        return x


class pytorch_model(nn.Module):
    def __init__(self, input_channels=3, print_shape=False, n_classes=81313):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 1)
        self.conv2 = nn.Conv2d(32, 8, 1)
        self.linear1 = nn.Linear(1152, n_classes)
        self.arc_face = ArcMarginProduct(1152, n_classes, s=30, m=0.5)

    def forward(self, x):
        x, y = x
        x = self.conv1(x.float())
        x = F.relu(x)
        x = nn.MaxPool2d(4, 4)(x)
        x = F.relu(self.conv2(x).float())
        x = nn.MaxPool2d(4, 4)(x)
        x = nn.Flatten()(x)
        x = self.arc_face(x, y)
        return x


def pytorch_train_loop(dataloader, model, loss_fn, optimizer, writer, epoch, device):
    size = dataloader.dataset.number_of_images
    correct, running_loss = 0, 0.0

    for batch, (X, y) in enumerate(tqdm(dataloader)):
        # Inference
        y = y[:, 0].to(device)

        X = X.permute(0, 4, 2, 3, 1).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
        X = X[:, :, :, :, 0]  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
        y_pred = model((X, y))

        # Calculate loss function
        loss = loss_fn(y_pred, y)
        y_pred_temp = torch.argmax(torch.Tensor.detach(y_pred), dim=1)
        correct += (np.round(torch.Tensor.cpu(y_pred_temp)) == torch.Tensor.cpu(y)).type(torch.float).sum().item()

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and save metrics
        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            writer.add_scalar('training_loss',
                              running_loss / 1000,
                              epoch * len(dataloader) + batch)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    writer.add_scalar('training_accuracy', correct, epoch + 1)
    print(f"Train Error:  Accuracy: {(100 * correct):>0.1f}%\n")


def pytorch_test_loop(dataloader, model, loss_fn, writer, epoch, device):
    size = dataloader.dataset.number_of_images
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(dataloader)):
            # Inference
            y = y[:, 0].to(device)
            X = X.permute(0, 4, 2, 3, 1).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
            X = X[:, :, :, :, 0]  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
            y_pred = model((X, y))
            test_loss += loss_fn(y_pred, y).item()
            y_pred_temp = torch.argmax(torch.Tensor.detach(y_pred), dim=1)
            correct += (np.round(torch.Tensor.cpu(y_pred_temp)) == torch.Tensor.cpu(y)).type(torch.float).sum().item()

    # Calculate and save metrics
    writer.add_scalar('test_loss', test_loss, epoch)
    test_loss /= num_batches
    correct /= size
    writer.add_scalar('test_accuracy', correct, epoch + 1)
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def inference_from_similarity(cosine_similarities, train_labels, k=3):
    # Predict labels based on similarity

    # Find k biggest similarities and the corresponding labels
    k_similar_indices = np.argpartition(cosine_similarities, -k, axis=1)[:, -k:]
    k_similar_labels = []
    k_biggest_similarities = []
    for i in range(k):
        index = k_similar_indices[0, i]
        k_similar_labels.append(train_labels[index])
        k_biggest_similarities.append(cosine_similarities[0, index])

    # Soft voting
    unique_labels = list(np.unique(k_similar_labels))
    voting_dict = {label: 0 for label in unique_labels}
    for label, similarity in zip(k_similar_labels, k_biggest_similarities):
        voting_dict[label] += similarity
    return max(voting_dict.items(), key=itemgetter(1))[0]


def pytorch_embedding_test(training_dataloader, validation_dataloader, model, writer, epoch, device, k=3):
    # Training embeddings
    train_labels = []
    train_embeddings = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(training_dataloader)):
            # Extract feature embeddings
            y = y[:, 0].to(device)
            X = X.permute(0, 4, 2, 3, 1).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
            X = X[:, :, :, :, 0]  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
            temp_embeddings = model.extract_features(X)
            temp_embeddings = np.array(temp_embeddings.cpu())
            y = np.array(y.cpu())

            # Add to lists
            for i in temp_embeddings:
                train_embeddings.append(i)
            for i in y:
                train_labels.append(i)

    # Test embeddings
    test_labels = []
    test_embeddings = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(tqdm(validation_dataloader)):
            # Inference
            y = y[:, 0].to(device)
            X = X.permute(0, 4, 2, 3, 1).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
            X = X[:, :, :, :, 0]  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
            temp_embeddings = model.extract_features(X)
            temp_embeddings = np.array(temp_embeddings.cpu())
            y = np.array(y.cpu())

            # Add to lists
            for i in temp_embeddings:
                test_embeddings.append(i)
            for i in y:
                test_labels.append(i)

    # Cosine similarity and class prediction
    predictions = []
    batch_size = 64
    test_embeddings_dataloader = DataLoader(test_embeddings, batch_size=batch_size)
    for batch in tqdm(test_embeddings_dataloader):
        # Calculate cosine similarities
        cosine_similarities = cosine_similarity(batch, train_embeddings)

        for j in range(batch.shape[0]):
            cosine_similarities_temp = cosine_similarities[j, :].reshape(1, -1)

            # Find k biggest similarities and the corresponding labels
            k_similar_indices = np.argpartition(cosine_similarities_temp, -k, axis=1)[:, -k:]
            k_similar_labels = []
            k_biggest_similarities = []
            for i in range(k):
                index = k_similar_indices[0, i]
                k_similar_labels.append(train_labels[index])
                k_biggest_similarities.append(cosine_similarities_temp[0, index])

            # Soft voting
            unique_labels = list(np.unique(k_similar_labels))
            voting_dict = {label: 0 for label in unique_labels}
            for label, similarity in zip(k_similar_labels, k_biggest_similarities):
                voting_dict[label] += similarity
            predictions.append(max(voting_dict.items(), key=itemgetter(1))[0])

    # Calculate Accuracy and F1
    accuracy = accuracy_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions, average='micro')
    writer.add_scalar('embedding_test_accuracy', accuracy, epoch + 1)
    print(f"Embedding Test Error: Accuracy: {(100 * accuracy):>0.1f}% \n, F1: {f1}")
    return 0


def preprocess_images(dir0, path, img_size, validation_set):
    images_succesfully_saved = 0
    directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    for dir1 in directories:
        for dir2 in directories:
            temp_path = path + '/train/' + dir0 + '/' + dir1 + '/' + dir2
            images = os.listdir(temp_path)
            for image_name in images:
                # Read and resize image
                img_array = cv2.imread(temp_path + '/' + image_name)  # / 255.0
                img_array = cv2.resize(img_array, (img_size, img_size))

                # cv2.imshow('image_name', img_array)
                # cv2.waitKey(0)

                # Check if it image is in the validation set
                directory = '/training_set/'
                if image_name[:-4] in validation_set:
                    directory = '/validation_set/'

                # Write image
                image_saved = cv2.imwrite(path + directory + image_name, img_array)
                if image_saved:
                    images_succesfully_saved += 1
                else:
                    print(f"Image not saved: {image_name}")
    return images_succesfully_saved


def preprocess_data(path, img_size=175, validation_size=0.25, classes=81313):
    directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    labels = dict(pd.read_csv(filepath_or_buffer=path + '/train.csv').values)
    keys = list(labels.keys())
    values = list(labels.values())
    unique_classes = set(labels.values())

    # Data will be passed to each dictionary
    # Then the dictionaries will be converted to dataframes
    # This drastically improves execution time
    validation_dict = {}
    training_dict = {}

    # Sample validation set using stratification
    validation_set = []
    dictionary_index = 0

    # Find the samples that are in each class
    for class_ in tqdm(unique_classes):
        class_samples = []
        while True:
            if values[dictionary_index] == class_:
                class_samples.append((keys[dictionary_index], values[dictionary_index]))
                dictionary_index += 1
            else:
                break
            if dictionary_index == len(values):
                break

        # Add a percentage of each classes samples in the validation set
        number_of_samples = len(class_samples)
        validation_samples = sample(class_samples, math.floor(number_of_samples * validation_size))

        # # Remove validation samples from the class samples
        training_samples = deepcopy(class_samples)
        [training_samples.remove(i) for i in validation_samples]

        for val_sample in validation_samples:
            validation_set.append(val_sample[0])
            validation_dict[val_sample[0]] = val_sample[1]

        for training_sample in training_samples:
            training_dict[training_sample[0]] = training_sample[1]

    # Delete variables to save memory
    del values
    del keys
    del labels
    del unique_classes

    # Convert dictionaries to dataframes
    validation_df = pd.DataFrame.from_dict(validation_dict, orient='index')
    del validation_dict
    training_df = pd.DataFrame.from_dict(training_dict, orient='index')
    del training_dict

    total_dataframe_samples = len(validation_df) + len(training_df)
    print(f"Total dataframe samples = {total_dataframe_samples}")

    # Save Dataframes to csv
    validation_df.to_csv(path + '/validation_dataframe.csv', index=True, header=False)
    del validation_df
    training_df.to_csv(path + '/training_dataframe.csv', index=True, header=False)
    del training_df

    # Read images, resize them and save them in new directories
    images_succesfully_saved = 0
    results = Parallel(n_jobs=8, prefer="threads")(
        delayed(preprocess_images)(dir, path, img_size, validation_set) for dir in directories)
    print(results)
    for i in results:
        images_succesfully_saved += i
    print(f"Images successfully saved: {images_succesfully_saved}")
    print("done")
    return


class CustomDataset(Dataset):
    def __init__(self, batch_size, data_path, labels_dataframe_path, IMG_SIZE, unique_classes,
                 is_validation_dataset=False):
        self.batch_size = batch_size
        self.data_path = data_path
        self.IMG_SIZE = IMG_SIZE
        self.labels = dict(pd.read_csv(filepath_or_buffer=labels_dataframe_path, header=None).values)
        self.unique_classes = unique_classes
        self.is_validation_sequence = is_validation_dataset

        self.current_dir_file_list = os.listdir(data_path)
        self.number_of_images = len(self.current_dir_file_list)


    def __len__(self):
        return math.ceil(self.number_of_images / self.batch_size)

    def __getitem__(self, index):
        # Calculate start index and end index
        start_index = index * self.batch_size
        if start_index + self.batch_size < self.number_of_images:
            end_index = start_index + self.batch_size
        else:
            end_index = self.number_of_images

        batch_lenth = end_index - start_index
        X = np.zeros((batch_lenth, self.IMG_SIZE, self.IMG_SIZE, 3))
        y = np.zeros((batch_lenth,))

        # Read batch and append it to batch list
        for in_batch_index, image_index in enumerate(range(start_index, end_index)):
            image_name = self.current_dir_file_list[image_index]
            # Read image and scale
            img_array = cv2.imread(self.data_path + '/' + image_name) / 255.0

            # cv2.imshow('image_name', img_array)
            # cv2.waitKey(0)

            # Remove the last 4 characters (.png) and get the label from the dictionary
            y_temp = self.labels[image_name[:-4]]

            # Append to batch
            X[in_batch_index] = img_array
            y[in_batch_index] = y_temp

        y_one_hot = label_binarize(y, classes=self.unique_classes)
        y = np.argmax(y_one_hot, axis=1)
        return torch.tensor(X), torch.tensor(y)
