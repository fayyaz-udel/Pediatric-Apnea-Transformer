import pickle

import numpy as np
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from metrics import Result

MODEL_NAME = "GB"
DATA_PATH = "C:\\Data\\filtered_balanced.npz"
FOLD = 5
THRESHOLD = 1

def select_model(model_name):
    if model_name == "GB":
        return GradientBoostingClassifier()
    if model_name == "LR":
        return LogisticRegression()
    if model_name == "SVM":
        return svm.SVC()
    if model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(128, 32, 8))


if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    ########################################################################################
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])

        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, :4]
    ########################################################################################
    for fold in range(FOLD):
        first = True
        for i in range(FOLD):
            if i == fold:
                x_test = x[i]
                y_test = y[i]
            else:
                if first:
                    x_train = x[i]
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, x[i]))
                    y_train = np.concatenate((y_train, y[i]))

    x_train_flatten = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    x_test_flatten = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] * x_test.shape[2]))

    result = Result()

    model = select_model(MODEL_NAME)
    model = model.fit(x_train_flatten, y_train)

    y_predict = model.predict(x_test_flatten)
    y_score = model.predict_proba(x_test_flatten)[:, 1]
    result.add(y_test, y_predict, y_score)

result.print()
