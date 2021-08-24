"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
import numpy as np
import pandas as pd
from random import shuffle, seed, sample
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score
import cv2
import math
import glob
import os
import time
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from torch.utils.tensorboard import SummaryWriter
from efficientnet_pytorch import EfficientNet


class PytorchTransferModel(nn.Module):
    def __init__(self, input_channels=3, print_shape=False, n_classes=81313):
        super().__init__()
        # self.model = vgg16(pretrained=True)
        #
        # # Freeze parameters
        # for param in self.model.parameters():
        #     param.requires_grad = False
        #
        # # Add final layer
        # num_features = self.model.classifier._modules['6'].in_features
        # self.model.classifier._modules['6'] = nn.Linear(num_features, n_classes)
        # self.model.classifier._modules['7'] = nn.Softmax(dim=1) #F.softmax(dim=1)

        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        for param in self.model.parameters():
            param.requires_grad = False
        # del self.model._modules['_swish']
        # del self.model._modules['_fc']
        self.fc = nn.Linear(5120, n_classes)

        fet = 5

    def forward(self, x):
        x = self.model.extract_features(x.float())
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.Flatten()(x)
        print(x.shape)
        x = F.softmax(self.fc(x), dim=1)

        return x


class pytorch_model(nn.Module):
    def __init__(self, input_channels=3, print_shape=False, n_classes=81313):
        super().__init__()
        # self.convolutional = nn.Sequential(
        #     nn.Conv2d(input_channels, 32, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Conv2d(32, 64, 5),
        #     nn.ReLU(),
        #     nn.MaxPool2d((2, 2)),
        #     nn.Flatten(),
        #     nn.Linear(5184, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 81313),
        #     nn.Softmax()
        # )
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.conv2 = nn.Conv2d(32, 8, 5)
        self.linear1 = nn.Linear(32, n_classes)

    def forward(self, x):
        # x = self.convolutional(x)
        x = self.conv1(x.float())
        x = F.relu(x)
        x = nn.MaxPool2d(8, 8)(x)
        x = F.relu(self.conv2(x).float())
        x = nn.MaxPool2d(8, 8)(x)
        x = nn.Flatten()(x)
        # print(x.shape)
        x = F.softmax(self.linear1(x), dim=1)

        return x


def pytorch_train_loop(dataloader, model, loss_fn, optimizer, writer, epoch, device):
    size = dataloader.number_of_images
    num_batches = len(dataloader)
    correct, running_loss = 0, 0.0

    for batch in tqdm(range(num_batches)):
        # print(f"Batch = {batch}")
        # Load Data
        X, y = dataloader[batch]

        # Compute prediction and loss
        y = y.to(device)
        X = X.permute(0, 3, 1, 2).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
        y_pred = model(X)  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
        loss = loss_fn(y_pred, y)
        y_pred_temp = torch.argmax(torch.Tensor.detach(y_pred), dim=1)
        correct += (np.round(torch.Tensor.cpu(y_pred_temp)) == torch.Tensor.cpu(y)).type(torch.float).sum().item()

        # Back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            writer.add_scalar('training_loss',
                              running_loss / 1000,
                              epoch * len(dataloader) + batch)

            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    correct /= size
    writer.add_scalar('training_accuracy',
                      correct,
                      epoch + 1)
    print(f"Train Error:  Accuracy: {(100 * correct):>0.1f}%\n")


def pytorch_test_loop(dataloader, model, loss_fn, writer, epoch, device):
    size = dataloader.number_of_images
    num_batches = len(dataloader)
    test_loss, correct = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(range(num_batches)):
            # Load Data
            X, y = dataloader[batch]

            y = y.to(device)
            X = X.permute(0, 3, 1, 2).to(device)  # Permute from (Batch_size,IMG_SIZE,IMG_SIZE,CHANNELS)
            y_pred = model(X)  # To (Batch_size,CHANNELS,IMG_SIZE,IMG_SIZE)
            test_loss += loss_fn(y_pred, y).item()
            y_pred_temp = torch.argmax(torch.Tensor.detach(y_pred), dim=1)
            correct += (np.round(torch.Tensor.cpu(y_pred_temp)) == torch.Tensor.cpu(y)).type(torch.float).sum().item()

    writer.add_scalar('test_loss', test_loss, epoch)

    test_loss /= num_batches
    correct /= size
    writer.add_scalar('test_accuracy',
                      correct,
                      epoch + 1)
    print(f"Test Error: Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


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

    # print(len(values))

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
    for dir0 in tqdm(directories):
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
    print(f"Images successfully saved: {images_succesfully_saved}")
    return


class DataLoader(Dataset):
    def __init__(self, batch_size, data_path, labels_dataframe_path, IMG_SIZE, unique_classes,
                 is_validation_sequence=False):
        self.batch_size = batch_size
        self.data_path = data_path
        self.IMG_SIZE = IMG_SIZE
        self.labels = dict(pd.read_csv(filepath_or_buffer=labels_dataframe_path, header=None).values)
        self.unique_classes = unique_classes
        self.is_validation_sequence = is_validation_sequence

        self.current_dir_file_list = os.listdir(data_path)
        self.number_of_images = len(self.current_dir_file_list)

        # TODO
        # if not self.is_validation_sequence:
        # self.number_of_images = 10 * self.batch_size

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
            # print(idx)
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
