import keras
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

from data_generator import get_all_data
from models import create_cnn_model


def lr_schedule(epoch, lr):
    if epoch > 20 and (epoch - 1) % 5 == 0:
        lr *= 0.5
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    X, y = get_all_data(r"C:\Data\preprocessed_old")
    y = keras.utils.to_categorical(y, num_classes=2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

    ACC = []
    SN = []
    SP = []
    F2 = []

for train, test in kfold.split(X, y.argmax(1)):
    model = create_cnn_model()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopper = EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(x=X[train], y=y[train], batch_size=128, epochs=100, validation_split=0.1,
                        callbacks=[early_stopper, lr_scheduler])

    loss, accuracy = model.evaluate(X[test], y[test])
    y_score = model.predict(X[test])
    y_predict = np.argmax(y_score, axis=-1)
    y_training = np.argmax(y[test], axis=-1)

    # Confusion matrix:
    C = confusion_matrix(y_training, y_predict, labels=(1, 0))
    TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
    acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
    f1 = f1_score(y_training, y_predict)

    ACC.append(acc * 100)
    SN.append(sn * 100)
    SP.append(sp * 100)
    F2.append(f1 * 100)

print("Accuracy: %.2f -+ %.3f" % (np.mean(ACC), np.std(ACC)))
print("Sensitivity: %.2f -+ %.3f" % (np.mean(SN), np.std(SN)))
print("Specifity: %.2f -+ %.3f" % (np.mean(SP), np.std(SP)))
print("F1: %.2f -+ %.3f" % (np.mean(F2), np.std(F2)))