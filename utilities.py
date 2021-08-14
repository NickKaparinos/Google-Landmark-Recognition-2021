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
from random import shuffle
from sklearn.model_selection import train_test_split
import cv2
import math
import glob
import os
import time


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


def training_epoch(data_loader, model, classes=81313):
    i = 0
    for x, y in data_loader.return_data():
        print(i)
        y_one_hot = np.array(tf.one_hot(y, classes))
        model.fit(x, y_one_hot)
        i += 1


class DataLoader(Sequence):
    def __init__(self, batch_size, data_path, IMAGE_SIZE, classes=81313):
        self.batch_size = batch_size
        self.train_path = data_path + '/train'
        self.file_index = 0
        self.IMAGE_SIZE = IMAGE_SIZE
        self.labels = dict(pd.read_csv(filepath_or_buffer=data_path + '/train.csv').values)
        self.classes = classes

        # Calculate 3 directory permutations
        # Each epoch the directories must be acces in a random order
        self.directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        self.permutations = self.calc_permutations()

        # Calculate current directory and its file list
        self.current_dir = [self.permutations[i][0] for i in range(3)]
        self.current_dir_file_list = os.listdir(
            self.train_path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[2])

        # Permute image list too
        self.current_dir_file_list = list(np.random.permutation(self.current_dir_file_list))

    def calc_permutations(self):
        return [np.random.permutation(self.directories) for _ in range(3)]

    def make_path(self):
        pass

    def __len__(self):
        # return math.ceil(len(self.labels) / self.batch_size)
        return  math.ceil(382 / self.batch_size)

    def __getitem__(self, index):
        # print(f"Mphka {index}")
        batch = []
        # X = None
        # y_one_hot = None
        # for dir0 in self.permutations[0]:
        # for dir1 in self.permutations[1]:
        # for dir2 in self.permutations[2]:
        start_index = index * self.batch_size
        print(self.current_dir)
        if start_index + self.batch_size <= len(self.current_dir_file_list):
            end_index = start_index + self.batch_size
        else:
            end_index = len(self.current_dir_file_list)

        for idx in range(start_index, end_index):
            # print(idx)
            image_name = self.current_dir_file_list[idx]
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

        x = np.array([b[0] for b in batch])
        y = np.array([b[1] for b in batch])
        y_one_hot = np.array(tf.one_hot(y, self.classes))
        end = time.perf_counter()
        return x, y_one_hot

