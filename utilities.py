"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.metrics import Precision, Recall
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import summary
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import Callback
from tensorflow.keras.utils import Sequence
# import efficientnet.keras as efn
import tensorflow.keras.layers as L
from tensorflow.keras.utils import Sequence
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
import efficientnet.tfkeras as efn


def build_model(input_shape, classes=81313):
    # model_backbone = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape)
    model_backbone = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)

    # Freeze layers
    for layer in model_backbone.layers:
        layer.trainable = False

    # Add backbone
    model = Sequential()
    model.add(model_backbone)
    # model = Sequential(model.layers)

    # Add neck
    model.add(Flatten())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall()])
    return model


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


class DataSequence(Sequence):
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
        # self.number_of_images = 100 * self.batch_size

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
        return X, y_one_hot
