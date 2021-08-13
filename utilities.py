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


# class DataGenerator(Sequence):
#     def __init__(self, path, list_IDs, data, img_size, img_channel, batch_size):
#         self.path = path
#         self.list_IDs = list_IDs
#         self.data = data
#         self.img_size = img_size
#         self.img_channel = img_channel
#         self.batch_size = batch_size
#         self.indexes = np.arange(len(self.list_IDs))
#
#     def __len__(self):
#         len_ = int(len(self.list_IDs)/self.batch_size)
#         if len_*self.batch_size < len(self.list_IDs):
#             len_ += 1
#         return len_
#
#     def __getitem__(self, index):
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
#         X, y = self.__data_generation(list_IDs_temp)
#         return X, y
#
#
#     def __data_generation(self, list_IDs_temp):
#         X = np.zeros((self.batch_size, self.img_size, self.img_size, self.img_channel))
#         y = np.zeros((self.batch_size, 1), dtype=int)
#         for i, ID in enumerate(list_IDs_temp):
#
#             image_id = self.data.loc[ID, 'id']
#             file = image_id+'.jpg'
#             subpath = '/'.join([char for char in image_id[0:3]])
#
# #             print(self.path+subpath+'/'+file)
#             img = cv2.imread(self.path+subpath+'/'+file)
# #             print(img)
#             img = img/255
#             img = cv2.resize(img, (self.img_size, self.img_size))
#             X[i, ] = img
#             if self.path.find('train')>=0:
#                 y[i, ] = self.data.loc[ID, 'landmark_id']
#             else:
#                 y[i, ] = 0
#         return X, y



class DataLoader:
    def __init__(self, batch_size, path):
        self.batch_size = batch_size
        self.path = path
        self.file_index = 0

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

        fet = 5

    def calc_permutations(self):
        return [np.random.permutation(self.directories) for _ in range(3)]

    def make_path(self):
        pass

    def return_data(self):
        for image in self.current_dir_file_list:
            img = cv2.imread(self.path + '/' + self.current_dir[0] + '/' + self.current_dir[1] + '/' + self.current_dir[2] + '/' + image)
            cv2.imshow('image', img)
            cv2.waitKey(0)


        return 5
