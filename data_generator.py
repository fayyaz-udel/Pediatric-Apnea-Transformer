import os
from random import shuffle
import numpy as np
import keras
from biosppy.signals.ecg import correct_rpeaks, hamilton_segmenter
from scipy.interpolate import splev, splrep
from scipy.signal import medfilt

SIGNAL_LENGTH = 3000
SIGNAL_SCALE = 50000
IN_FREQ = 100
CHANNELS_NO = 4
OUT_FREQ = 3


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

    X = np.empty((length, 90, CHANNELS_NO))
    y = np.empty(length, dtype=float)

    bad_idx = []
    scaler = lambda arr: arr  # * SIGNAL_SCALE #(arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    ir = 3  # INTERPOLATION RATE(3HZ)
    tm = np.arange(0, 30, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    for i in range(length):
        if i % 1000 == 0:
            print("Read sample: " + str(i))
        path = root_dir + '/' + file_list[i]
        tmp = np.load(path)

        signal = np.squeeze(tmp['data'][0]) * SIGNAL_SCALE
        rpeaks, = hamilton_segmenter(signal, sampling_rate=IN_FREQ)
        rpeaks, = correct_rpeaks(signal, rpeaks=rpeaks, sampling_rate=IN_FREQ, tol=0.1)

        if len(rpeaks) < 15 or len(rpeaks) > 100:
            print(len(rpeaks))
            bad_idx.append(i)
            continue
        rri_tm, rri_signal = rpeaks[1:] / float(IN_FREQ), np.diff(rpeaks) / float(IN_FREQ)
        # rri_signal = medfilt(rri_signal, kernel_size=3)
        ampl_tm, ampl_signal = rpeaks / float(IN_FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, scaler(rri_signal), k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, scaler(ampl_signal), k=3), ext=1)
        X[i, :, 0] = rri_interp_signal
        X[i, :, 1] = amp_interp_signal
        X[i, :, 2] = np.interp(np.arange(0, SIGNAL_LENGTH, 33.34), np.arange(0, SIGNAL_LENGTH),
                                                 tmp['data'][1]) * SIGNAL_SCALE
        X[i, :, 3] = np.interp(np.arange(0, SIGNAL_LENGTH, 33.34), np.arange(0, SIGNAL_LENGTH),
                                                 tmp['data'][2]) - 80

        # X[i, :, 0] = np.squeeze(tmp['data']) * SIGNAL_SCALE #(np.squeeze((tmp['data'] - tmp['data'].min()) / (tmp['data'].max() - tmp['data'].min())))
        y[i] = tmp['labels']

    return np.delete(X, bad_idx, 0), np.delete(y, bad_idx, 0)


if __name__ == "__main__":
    get_all_data(r"C:\Data\preprocessed_three")
