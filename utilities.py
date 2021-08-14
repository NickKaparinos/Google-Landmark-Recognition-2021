"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import Sequence
# import efficientnet.keras as efn
import tensorflow.keras.layers as L
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from random import shuffle, seed, sample
from sklearn.model_selection import train_test_split
import cv2
import math
import glob
import os
import time
from tqdm import tqdm


def returnVGG16(input_shape, classes=81313):
    model = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape, classes=2,
                                        classifier_activation='sigmoid')
    for layer in model.layers:
        layer.trainable = False

    model = Sequential(model.layers)
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def training(data_loader, model, classes=81313, epochs=1):
    i = 0
    for _ in range(epochs):
        for x, y in data_loader.generate_date():
            # print(i)
            model.fit(x, y)
            i += 1


# class DataLoader(Sequence):
#     def __init__(self, batch_size, data_path, IMAGE_SIZE, classes=81313):
#         self.batch_size = batch_size
#         self.train_path = data_path + '/train'
#         self.file_index = 0
#         self.IMAGE_SIZE = IMAGE_SIZE
#         self.labels = dict(pd.read_csv(filepath_or_buffer=data_path + '/train.csv').values)
#         self.classes = classes
#         self.index = 0
#
#         # Calculate 3 directory permutations
#         # Each epoch the directories must be acces in a random order
#         self.directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
#         self.permutations = self.calc_permutations()
#
#         # Calculate current directory and its file list
#         self.current_dir = [self.permutations[i][0] for i in range(3)]
#         self.current_dir_file_list = self.get_current_dir_file_list()
#
#         # Permute image list too
#         self.current_dir_file_list = list(np.random.permutation(self.current_dir_file_list))
#
#     def calc_permutations(self):
#         return [np.random.permutation(self.directories) for _ in range(3)]
#
#     def make_path(self):
#         pass
#
#     def get_current_dir_file_list(self):
#         return os.listdir(self.train_path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[2])
#
#     def __len__(self):  # TODO
#         # return math.ceil(len(self.labels) / self.batch_size)
#         return  math.ceil(382 / self.batch_size)
#
#     def __getitem__(self, index):
#         self.index += 1
#         # print(f"Mphka {index}")
#         batch = []
#
#          # TODO
#         # for dir0 in self.permutations[0]:
#         # for dir1 in self.permutations[1]:
#         for dir2 in self.permutations[2]:
#             self.current_dir[2] = dir2
#             self.current_dir_file_list = self.get_current_dir_file_list()
#
#             start_index = self.index * self.batch_size
#             print(self.current_dir)
#
#             # Maybe delete ?
#             if start_index + self.batch_size <= len(self.current_dir_file_list):
#                 end_index = start_index + self.batch_size
#             else:
#                 end_index = len(self.current_dir_file_list)
#
#             for idx in range(start_index, end_index):
#                 # print(idx)
#                 image_name = self.current_dir_file_list[idx]
#                 # Read image and scale
#                 img_array = cv2.imread(
#                     self.train_path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[
#                         2] + '/' + image_name) / 255.0
#                 img_array = cv2.resize(img_array, (self.IMAGE_SIZE, self.IMAGE_SIZE))
#
#                 # cv2.imshow('image_name', img_array)
#                 # cv2.waitKey(0)
#
#                 # Remove the last 4 characters (.png) and get the label from the dictionary
#                 y = self.labels[image_name[:-4]]
#
#                 # Append to batch
#                 batch.append((img_array, y))
#
#         x = np.array([b[0] for b in batch])
#         y = np.array([b[1] for b in batch])
#         y_one_hot = np.array(tf.one_hot(y, self.classes))
#         # print(f"Bghka {index}")
#         return x, y_one_hot
#
#     def on_epoch_end(self):
#         print("kappa")
#     # TODO one epoch end, shuffle

def preprocess_data(path, IMG_SIZE=150, validation_size=0.25, classes=81313):
    directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
    labels = dict(pd.read_csv(filepath_or_buffer=path + '/train.csv').values)
    keys = list(labels.keys())
    values = list(labels.values())
    all_classes = set(labels.values())

    print(len(values))

    # Sample validation set using stratification
    validation_set = []
    dictionary_index = 0

    # Find the samples that are in each class
    for c in tqdm(all_classes):
        class_samples = []
        while (True):
            if values[dictionary_index] == c:
                class_samples.append(keys[dictionary_index])
                dictionary_index += 1
            else:
                break
            if dictionary_index == len(values) - 1:
                break

        # Add a percentage of each classes samples in the validation set
        number_of_samples = len(class_samples)
        validation_samples = sample(class_samples, math.floor(number_of_samples * validation_size))
        for val_sample in validation_samples:
            validation_set.append(val_sample)

    for dir0 in tqdm(directories):
        for dir1 in tqdm(directories):
            for dir2 in tqdm(directories):
                temp_path = path + '/train/' + dir0 + '/' + dir1 + '/' + dir2
                images = os.listdir(temp_path)
                for image_name in images:
                    img_array = cv2.imread(temp_path + '/' + image_name)  # / 255.0
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    # cv2.imshow('image_name', img_array)
                    # cv2.waitKey(0)

                    dir = '/training_set/'
                    if image_name[:-4] in validation_set:
                        dir = '/validation_set/'

                    cv2.imwrite(path + dir + image_name, img_array)
    return


class DataLoader():
    def __init__(self, batch_size, data_path, IMAGE_SIZE, classes=81313):
        self.batch_size = batch_size
        self.train_path = data_path + '/train'
        self.file_index = 0
        self.IMAGE_SIZE = IMAGE_SIZE
        self.labels = dict(pd.read_csv(filepath_or_buffer=data_path + '/train.csv').values)
        self.classes = classes
        self.index = 0

        # Calculate 3 directory permutations
        # Each epoch the directories must be acces in a random order
        self.directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        self.permutations = self.calc_permutations()

        # Calculate current directory and its file list
        self.current_dir = [self.permutations[i][0] for i in range(3)]
        self.current_dir_file_list = self.get_current_dir_file_list()

        # Permute image list too
        self.current_dir_file_list = list(np.random.permutation(self.current_dir_file_list))

    def calc_permutations(self):
        return [np.random.permutation(self.directories) for _ in range(3)]

    def make_path(self):
        pass

    def get_current_dir_file_list(self):
        return os.listdir(
            self.train_path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[2])

    def __len__(self):  # TODO
        # return math.ceil(len(self.labels) / self.batch_size)
        return math.ceil(382 / self.batch_size)

    def generate_date(self):
        # print(f"Mphka {index}")
        batch = []

        # TODO
        # for dir0 in self.permutations[0]:
        # for dir1 in self.permutations[1]:
        # for dir2 in self.permutations[2]:
        #     self.current_dir[2] = dir2
        self.current_dir_file_list = self.get_current_dir_file_list()

        start_index = self.index * self.batch_size
        print(self.current_dir)

        # Maybe delete ?
        if start_index + self.batch_size <= len(self.current_dir_file_list):
            end_index = start_index + self.batch_size
        else:
            end_index = len(self.current_dir_file_list)

        for idx, image_name in enumerate(self.current_dir_file_list):
            # print(f"image idx {idx}")
            # Read image and scale
            img_array = cv2.imread(
                self.train_path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[
                    2] + '/' + image_name) / 255.0
            img_array = cv2.resize(img_array, (self.IMAGE_SIZE, self.IMAGE_SIZE))

            # cv2.imshow('image_name', img_array)
            # cv2.waitKey(0)

            # Remove the last 4 characters (.png) and get the label from the dictionary
            y = self.labels[image_name[:-4]]

            # Append to batch
            batch.append((img_array, y))

            if len(batch) >= self.batch_size or idx == len(self.current_dir_file_list) - 1:  # TODO change or condit t
                x = np.array([b[0] for b in batch])
                y = np.array([b[1] for b in batch])
                y_one_hot = np.array(tf.one_hot(y, self.classes))
                # print(f"Bghka {index}")
                batch = []
                yield x, y_one_hot

    def on_epoch_end(self):
        print("kappa")
    # TODO one epoch end, shuffle
