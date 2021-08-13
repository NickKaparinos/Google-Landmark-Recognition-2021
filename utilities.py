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
# import efficientnet.keras as efn
import tensorflow.keras.layers as L
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
from random import shuffle
from sklearn.model_selection import train_test_split
import cv2
import math
import os
import time


def returnVGG16(input_shape):
    model = tf.keras.applications.VGG16(include_top=False, input_shape=input_shape, classes=2,
                                        classifier_activation='sigmoid')
    for layer in model.layers:
        layer.trainable = False

    model = Sequential(model.layers)
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(81313, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class DataLoader:
    def __init__(self, batch_size, path, IMAGE_SIZE):
        self.batch_size = batch_size
        self.path = path + '/train'
        self.file_index = 0
        self.IMAGE_SIZE = IMAGE_SIZE
        self.labels = dict(pd.read_csv(filepath_or_buffer=path + '/train.csv').values)

        # Calculate 3 directory permutations
        # Each epoch the directories must be acces in a random order
        self.directories = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        self.permutations = self.calc_permutations()

        # Calculate current directory and its file list
        self.current_dir = [self.permutations[i][0] for i in range(3)]
        self.current_dir_file_list = os.listdir(
            self.path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[2])

        # Permute image list too
        self.current_dir_file_list = list(np.random.permutation(self.current_dir_file_list))

    def calc_permutations(self):
        return [np.random.permutation(self.directories) for _ in range(3)]

    def make_path(self):
        pass

    def return_data(self):
        batch = []
        for image_name in self.current_dir_file_list:
            # Read image and scale
            img_array = cv2.imread(
                self.path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[
                    2] + '/' + image_name) / 255.0
            img_array = cv2.resize(img_array, (self.IMAGE_SIZE, self.IMAGE_SIZE))

            # cv2.imshow('image_name', img_array)
            # cv2.waitKey(0)

            # Remove the last 4 characters (.png) and get the label from the dictionary
            y = self.labels[image_name[:-4]]

            # Append to batch
            batch.append((img_array, y))

            if len(batch) >= self.batch_size:
                break

        x =  np.array([b[0] for b in batch])
        y =  np.array([b[1] for b in batch])
        return x,y
