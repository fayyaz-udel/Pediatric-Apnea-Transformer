import keras
from keras import layers
from config import *

def get_augmentation_model():
    model = keras.Sequential(
        [layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),],
        name="data_augmentation",
    )
    return model