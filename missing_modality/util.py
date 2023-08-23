import keras
import numpy as np
from keras import layers
from config import *
from scipy import signal

def get_augmentation_model():
    model = keras.Sequential(
        [layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),],
        name="data_augmentation",
    )
    return model


def transform2freq(x, idx):
    out_x = np.zeros((x.shape[0], 128, 16, 1))
    for i in range(x.shape[0]):
        f, t, Zxx = signal.stft(x[i, :, idx], fs=64, padded=False)
        Zxx = np.squeeze(Zxx)
        Zxx = np.abs(Zxx)[:128, :16]
        out_x[i, :, :, 0] = ((Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx)))
    return out_x