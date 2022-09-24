from keras.layers import Conv1D, MaxPooling1D, AveragePooling2D, AveragePooling1D
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, Permute, Reshape, LSTM, BatchNormalization, Bidirectional
from keras.models import Sequential


def create_cnn_model():
    model = Sequential()
    model.add(Conv1D(16, kernel_size=50, activation="relu"))
    model.add(MaxPooling1D(pool_size=10))
    model.add(Conv1D(32, kernel_size=20, activation="relu"))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Conv1D(64, kernel_size=10, activation="relu"))

    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(2, activation="softmax"))

    return model
