import keras
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, f1_score, average_precision_score, roc_auc_score

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

DATA_PATH = "C:\\Data\\filtered_balanced.npz"
THRESHOLD = 1
FOLD = 5 #TODO
if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
    ############################################################################

    ACC = []
    SN = []
    SP = []
    F1 = []
    AUC = []
    AUPRC = []

    for fold in range(FOLD):
        x_test = np.nan_to_num(x[fold], nan=-1)
        y_test = y[fold] # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)
        model = keras.models.load_model("./weightsbal/fold " + str(fold))

        y_score = sigmoid(model.predict(x_test))
        y_predict = np.where(y_score > 0.5, 1,0) # For MultiClass np.argmax(y_score, axis=-1)
        # For MultiClass y_test = np.argmax(y_test, axis=-1)

        f1 = f1_score(y_test, y_predict)
        auc = roc_auc_score(y_test, y_score) # For MultiClass y_score[:, 1]
        auprc = average_precision_score(y_test, y_score) # For MultiClass y_score[:, 1]

        # Confusion matrix:
        C = confusion_matrix(y_test, y_predict, labels=(1, 0))
        TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
        acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)

        ACC.append(acc * 100)
        SN.append(sn * 100)
        SP.append(sp * 100)
        F1.append(f1 * 100)
        AUC.append(auc * 100)
        AUPRC.append(auprc * 100)

    print(ACC)
    print(SN)
    print(SP)
    print(F1)
    print(AUC)
    print(AUPRC)

    print("Accuracy: %.2f -+ %.3f" % (np.mean(ACC), np.std(ACC)))
    print("Recall: %.2f -+ %.3f" % (np.mean(SN), np.std(SN)))
    print("Specifity: %.2f -+ %.3f" % (np.mean(SP), np.std(SP)))
    print("F1: %.2f -+ %.3f" % (np.mean(F1), np.std(F1)))
    print("AUROC: %.2f -+ %.3f" % (np.mean(AUC), np.std(AUC)))
    print("AUPRC: %.2f -+ %.3f" % (np.mean(AUPRC), np.std(AUPRC)))