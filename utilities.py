"""
Nick Kaparinos
Titanic - Machine Learning from Disaster
Kaggle Competition
"""
import numpy as np
import pandas as pd
import tensorflow as tf
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


class DataLoader:
    def __init__(self, batch_size, path, IMAGE_SIZE):
        self.batch_size = batch_size
        self.path = path
        self.file_index = 0
        self.IMAGE_SIZE = IMAGE_SIZE

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
            # Read image
            img_array = cv2.imread(
                self.path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[
                    2] + '/' + image_name)
            img_array = cv2.resize(img_array, (self.IMAGE_SIZE, self.IMAGE_SIZE))

            cv2.imshow('image_name', img_array)
            cv2.waitKey(0)

            # Append to batch
            batch.append((image_name, img_array))

            if len(batch) >= self.batch_size:
                break

        return batch
