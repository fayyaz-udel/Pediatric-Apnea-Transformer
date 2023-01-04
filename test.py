import keras
import numpy as np
from sklearn.utils import shuffle

from metrics import Result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


DATA_PATH = "C:\\Data\\filtered_balanced.npz"
MODEL_PATH = "./weightsbal/fold "
THRESHOLD = 1
FOLD = 1

if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
    ############################################################################
    result = Result()

    for fold in range(FOLD):
        x_test = np.nan_to_num(x[fold], nan=-1)
        y_test = y[fold]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)
        model = keras.models.load_model(MODEL_PATH + str(fold))

        y_score = sigmoid(model.predict(x_test) - 1.5)
        y_predict = np.where(model.predict(x_test) > 2, 1,0)  # np.where(y_score > 0.5, 1,0) # For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_test, y_predict, y_score)

    result.print()
