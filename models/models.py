from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, LSTM, Bidirectional, Permute, \
    Reshape, GRU, Conv1D, MaxPooling1D
from keras.models import Sequential
from keras.regularizers import l2

from .transformer import create_transformer_model


def create_ZFNet_BiLSTM_model(weight=1e-3):
    model = Sequential()
    model.add(Reshape((90, 2, 2), input_shape=(180, 1, 2)))
    model.add(Conv2D(96, kernel_size=(7, 1), strides=(1, 1), padding="same", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(256, kernel_size=(5, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(512, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(1024, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(3, 1), strides=(2, 1)))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((2, 4 * 512)))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(37, activation="relu"))

    model.add(Dense(2, activation="softmax"))
    return model


def create_ZFNet_model(weight=1e-3):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=(7), strides=(1), padding="valid", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 6)))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(BatchNormalization())
    model.add(
        Conv1D(64, kernel_size=(5), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(BatchNormalization())
    model.add(Conv1D(128, kernel_size=(31), strides=(1), padding="valid", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv1D(256, kernel_size=(3), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv1D(128, kernel_size=(3), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling1D(pool_size=(3), strides=(2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    return model


def create_VGG19_BiLSTM_model(weight=1e-3):
    model = Sequential()
    model.add(Reshape((90, 2, 2), input_shape=(180, 1, 2)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Permute((2, 1, 3)))
    model.add(Reshape((2, 5 * 512)))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(21, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def create_VGG19_model(weight=1e-3):
    model = Sequential()
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(Flatten())
    model.add(Dense(167, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


def create_VGG16_model(weight=1e-3):
    model = Sequential()

    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    return model


def create_VGG16_BiLSTM_model(weight=1e-3):
    model = Sequential()
    model.add(Reshape((90, 2, 2), input_shape=(180, 1, 2)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(64, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(128, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(
        Conv2D(512, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((2, 5 * 512)))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))

    model.add(Flatten())
    model.add(Dense(37, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    return model


def create_AlexNet_model(weight=1e-3):
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 1), strides=(1, 1), padding="valid", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(
        Conv2D(256, kernel_size=(5, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Conv2D(384, kernel_size=(3, 1), strides=(1, 1), padding="valid", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(384, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="valid", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1), strides=(2, 1)))

    model.add(Flatten())
    model.add(Dense(209, activation="relu"))
    model.add(Dense(2, activation="softmax"))
    return model


def create_AlexNet_BiLSTM_model(weight=1e-3):
    model = Sequential()
    model.add(Reshape((90, 2, 2), input_shape=(180, 1, 2)))
    model.add(Conv2D(96, kernel_size=(11, 1), strides=(1, 1), padding="same", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight), input_shape=(180, 1, 2)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(
        Conv2D(256, kernel_size=(5, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1)))
    model.add(Conv2D(384, kernel_size=(3, 1), strides=(1, 1), padding="same", activation="relu",
                     kernel_initializer="he_normal",
                     kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(384, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(
        Conv2D(256, kernel_size=(3, 1), strides=1, padding="same", activation="relu", kernel_initializer="he_normal",
               kernel_regularizer=l2(weight), bias_regularizer=l2(weight)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 1), strides=(2, 1)))

    model.add(Permute((2, 1, 3)))
    model.add(Reshape((2, 4 * 256)))

    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(18, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    return model


def create_BiLSTM_model():
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(180, 6)))
    model.add(LSTM(16, return_sequences=True))
    model.add(LSTM(8, return_sequences=True))

    model.add(Flatten())
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model


def create_GRU_model():
    model = Sequential()
    model.add(GRU(90, return_sequences=True, input_shape=(180, 6)))
    model.add(GRU(45, return_sequences=True))
    model.add(GRU(22, return_sequences=True))

    model.add(Flatten())

    model.add(Dense(7, activation="relu"))
    model.add(Dense(2, activation="sigmoid"))

    return model


model_dict = {
    "GRU": create_GRU_model(),
    "BiLSTM": create_BiLSTM_model(),
    "AlexNet": create_AlexNet_model(),
    "AlexNet_BiLSTM": create_AlexNet_BiLSTM_model(),
    # "ZFNet": create_ZFNet_model(),
}


def get_model(config):
    if config["model_name"].split('_')[0] == "Transformer":
        return create_transformer_model(input_shape=(60*32, len(config["channels"])),
                                        num_patches=config["num_patches"], projection_dim=config["transformer_units"],
                                        transformer_layers=config["transformer_layers"], num_heads=config["num_heads"],
                                        transformer_units=[config["transformer_units"] * 2,
                                                           config["transformer_units"]],
                                        mlp_head_units=[256, 128], num_classes=1, drop_out=config["drop_out_rate"],
                                        reg=config["regression"], l2_weight=config["regularization_weight"])
    else:
        return model_dict.get(config["model_name"].split('_')[0])


if __name__ == "__main__":
    config = {
        "model_name": "Transformer_129",
        "regression": False,

        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,
        "num_patches": 20,  # best
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best
        "channels": [0, 1,2,3,4,5],
    }
    model = get_model(config)
    print(model.summary())
