import os
from random import shuffle
import numpy as np
import keras

SIGNAL_LENGTH = 3000
SIGNAL_SCALE = 500


class DataGenerator(keras.utils.Sequence):
    def __init__(self, root_dir, batch_size=128):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.root_dir = os.path.expanduser(root_dir)
        self.file_list = os.listdir(self.root_dir)
        shuffle(self.file_list)

    def __len__(self):
        return int(np.floor(len(self.file_list) / self.batch_size))

    def __getitem__(self, index):
        X = np.empty((self.batch_size, SIGNAL_LENGTH, 1))
        y = np.empty(self.batch_size, dtype=float)

        counter = 0
        for i in range(index * self.batch_size, ((index + 1) * self.batch_size)):
            path = self.root_dir + '/' + self.file_list[i]
            tmp = np.load(path)
            X[counter, :, 0] = np.squeeze(tmp['data']) * SIGNAL_SCALE
            y[counter] = tmp['labels']
            counter += 1

        return X, y


def get_all_data(root_dir):
    root_dir = os.path.expanduser(root_dir)
    file_list = os.listdir(root_dir)
    shuffle(file_list)
    length = len(file_list)

    X = np.empty((length, SIGNAL_LENGTH, 1))
    y = np.empty(length, dtype=float)

    for i in range(length):
        if i % 1000 == 0:
            print("Read sample: " + str(i))
        path = root_dir + '/' + file_list[i]
        tmp = np.load(path)
        X[i, :, 0] = np.squeeze(tmp['data']) * SIGNAL_SCALE
        y[i] = tmp['labels']

    return X, y
