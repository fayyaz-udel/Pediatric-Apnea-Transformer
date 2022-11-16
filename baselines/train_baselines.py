from sklearn import svm
import numpy as np
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier


MODEL_NAME = "LR"


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
    with open(r'.\data\data.pkl', 'rb') as f:
        x, y = pickle.load(f)

    ACC = []
    SN = []
    SP = []
    F2 = []

for fold in range(5):

    first = True
    for i in range(5):
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

    model = select_model(MODEL_NAME)
    model = model.fit(x_train_flatten, y_train)

    y_predict = model.predict(x_test_flatten)
    C = confusion_matrix(y_test, y_predict, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_test, y_predict)

    ACC.append(acc * 100)
    SN.append(sn * 100)
    SP.append(sp * 100)
    F2.append(f1 * 100)

print("========================== " + MODEL_NAME + "==========================")
print(ACC)
print(SN)
print(SP)
print(F2)
print("Accuracy: %.2f -+ %.3f" % (np.mean(ACC), np.std(ACC)))
print("Sensitivity: %.2f -+ %.3f" % (np.mean(SN), np.std(SN)))
print("Specifity: %.2f -+ %.3f" % (np.mean(SP), np.std(SP)))
print("F1: %.2f -+ %.3f" % (np.mean(F2), np.std(F2)))
