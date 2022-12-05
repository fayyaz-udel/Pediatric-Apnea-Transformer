import pickle

import keras
from keras import metrics
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score

from models import create_vit_classifier

DATA_PATH = "C:\\Data\\filtered_bmi_age.npz"
THRESHOLD = 1


def lr_schedule(epoch, lr):
    if epoch > 50 and (epoch - 5) % 10 == 0:
        lr *= 0.25
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)

    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    for i in range(5):
        y[i] = np.greater_equal(y[i], THRESHOLD)
    ############################################################################

    ACC = []
    SN = []
    SP = []
    F2 = []

for fold in range(5):

    first = True
    for i in range(5):
        x[i] = x[i][:, :, [4]]
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

    y_train = keras.utils.to_categorical(y_train, num_classes=2)
    y_test = keras.utils.to_categorical(y_test, num_classes=2)

    model = create_vit_classifier()
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopper = EarlyStopping(patience=25, restore_best_weights=True)
    history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=100, validation_split=0.1,
                        callbacks=[early_stopper, lr_scheduler])

    loss, accuracy = model.evaluate(x_test, y_test)
    y_score = model.predict(x_test)
    y_predict = np.argmax(y_score, axis=-1)
    y_groundtruth = np.argmax(y_test, axis=-1)

    # Confusion matrix:
    C = confusion_matrix(y_groundtruth, y_predict, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_groundtruth, y_predict)

    ACC.append(acc * 100)
    SN.append(sn * 100)
    SP.append(sp * 100)
    F2.append(f1 * 100)

print(ACC)
print(SN)
print(SP)
print(F2)
print("Accuracy: %.2f -+ %.3f" % (np.mean(ACC), np.std(ACC)))
print("Sensitivity: %.2f -+ %.3f" % (np.mean(SN), np.std(SN)))
print("Specifity: %.2f -+ %.3f" % (np.mean(SP), np.std(SP)))
print("F1: %.2f -+ %.3f" % (np.mean(F2), np.std(F2)))
