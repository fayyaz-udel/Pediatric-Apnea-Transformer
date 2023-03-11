import keras.metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.utils import shuffle
from keras.losses import BinaryFocalCrossentropy, BinaryCrossentropy, Huber, MeanSquaredError, MeanAbsoluteError
import metrics
from metrics import Precision, Recall
from models.models import get_model

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

THRESHOLD = 1
FOLD = 5


def lr_schedule(epoch, lr):
    if epoch > 80 and (epoch - 1) % 10 == 0:
        lr *= 0.5
    # print("Learning rate: ", lr)
    return lr


def extract_features():
    pass


def train(config):
    data = np.load(config["data_path"], allow_pickle=True)
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    ########################################################################################
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        if config["regression"]:
            y[i] = np.sqrt(y[i])
            y[i][y[i] != 0] += 2
        else:
            y[i] = np.where(y[i] >= THRESHOLD, 1, 0)

        x[i] = x[i][:, :, config["channels"]]  # CHANNEL SELECTION

    ########################################################################################
    for fold in range(FOLD):
        first = True
        for i in range(5):
            if i != fold:
                if first:
                    x_train = x[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x[i]))
                    y_train = np.concatenate((y_train, y[i]))

        # y_train = keras.utils.to_categorical(y_train, num_classes=2) # For MultiClass
        # y_test = keras.utils.to_categorical(y_test, num_classes=2) # For MultiClass
        ################################################################################################################
        # Huber(), "mean_squared_error", "mean_absolute_error" Precision(from_logits=True), Recall(from_logits=True)

        model = get_model(config)
        if config["regression"]:
            model.compile(optimizer="adam", loss=BinaryCrossentropy())
            early_stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        else:
            model.compile(optimizer="adam", loss=BinaryCrossentropy(), metrics=[keras.metrics.Precision(), keras.metrics.Recall()])
            early_stopper = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

        lr_scheduler = LearningRateScheduler(lr_schedule)
        history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=config["epochs"], validation_split=0.1,
                            callbacks=[early_stopper, lr_scheduler])
        ################################################################################################################
        model.save(config["model_path"] + str(fold))
        keras.backend.clear_session()

