import glob
import os
import random
import numpy as np
import pandas as pd

SIGNAL_LENGTH = 3000
SIGNAL_SCALE = 50000
IN_FREQ = 100
CHANNELS_NO = 4
OUT_FREQ = 3
PATH = "C:\\Data\\p\\"


def load_data(path):
    age_bmi = pd.read_csv("../misc/result.csv")
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    study_event_counts = {}
    apnea_event_counts = {}
    hypopnea_event_counts = {}

    for i in range(length):
        patient_id = (file_list[i].split("_")[0])
        study_id = (file_list[i].split("_")[1])
        apnea_count = int((file_list[i].split("_")[2]))
        hypopnea_count = int((file_list[i].split("_")[3]).split(".")[0])

        apnea_event_counts[patient_id] = apnea_event_counts.get(patient_id, 0) + apnea_count
        hypopnea_event_counts[patient_id] = hypopnea_event_counts.get(patient_id, 0) + hypopnea_count
        study_event_counts[patient_id] = study_event_counts.get(patient_id, 0) + apnea_count + hypopnea_count

    apnea_event_counts = sorted(apnea_event_counts.items(), key=lambda item: item[1])
    hypopnea_event_counts = sorted(hypopnea_event_counts.items(), key=lambda item: item[1])
    study_event_counts = sorted(study_event_counts.items(), key=lambda item: item[1])

    folds = []
    for i in range(5):
        folds.append(study_event_counts[i::5])

    x = []
    y_apnea = []
    y_hypopnea = []
    counter = 0
    for fold in folds:
        first = True
        for patient in fold:
            counter += 1
            print(counter)
            for study in glob.glob(PATH + patient[0] + "_*"):
                study_data = np.load(study)

                signals = study_data['data']

                identifier = study.split('\\')[-1].split('_')[0] + "_" + study.split('\\')[-1].split('_')[1]
                bmi = np.ones((signals.shape[0], signals.shape[1])) * age_bmi[age_bmi['id'] == identifier].iat[0, 2] * 0.01
                age = np.ones((signals.shape[0], signals.shape[1])) * age_bmi[age_bmi['id'] == identifier].iat[0, 3] * 0.01

                data = np.zeros((signals.shape[0], signals.shape[1],6))
                data[:, :, 0:4] = signals
                data[:, :, 4] = bmi
                data[:, :, 5] = age
                labels_apnea = study_data['labels_apnea']
                labels_hypopnea = study_data['labels_hypopnea']
                if first:
                    aggregated_data = data
                    aggregated_label_apnea = labels_apnea
                    aggregated_label_hypopnea = labels_hypopnea
                    first = False
                else:
                    aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                    aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                    aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)

        x.append(aggregated_data)
        y_apnea.append(aggregated_label_apnea)
        y_hypopnea.append(aggregated_label_hypopnea)

    return x, y_apnea, y_hypopnea


def downsample(x, y_apnea, y_hypopnea, ratio=0.01):
    for f in range(5):
        print("down sampling fold " + str(f))
        x_c = x[f]
        y_apnea_c = y_apnea[f]
        y_hypopnea_c = y_hypopnea[f]
        y_c = y_apnea_c  # TODO

        neg_samples = np.where(y_c == 0)[0]
        pos_samples = list(np.where(y_c > 0)[0])

        neg_survived = []
        for s in range(len(neg_samples)):
            if random.random() < ratio:
                neg_survived.append(neg_samples[s])

        samples = neg_survived + pos_samples

        x[f] = x_c[samples, :, :]
        y_apnea[f] = y_apnea_c[samples]
        y_hypopnea[f] = y_hypopnea_c[samples]

    return x, y_apnea, y_hypopnea


if __name__ == "__main__":
    x, y_apnea, y_hypopnea = load_data(PATH)
    x, y_apnea, y_hypopnea = downsample(x, y_apnea, y_hypopnea)
    np.savez_compressed("C:\\Data\\filtered_bmi_age", x=x, y_apnea=y_apnea, y_hypopnea=y_hypopnea)
