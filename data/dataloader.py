import glob
import os

import numpy as np

SIGNAL_LENGTH = 3000
SIGNAL_SCALE = 50000
IN_FREQ = 100
CHANNELS_NO = 4
OUT_FREQ = 3


def load_data(path):
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    study_event_counts = {}
    apnea_event_counts = {}
    hypopnea_event_counts = {}

    for i in range(length):
        patient_id = (file_list[i].split("_")[0])
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
    for fold in folds:
        first = True
        for patient in fold:
            for study in glob.glob("C:\\Data\\processed\\" + patient[0] + "_*"):
                study_data = np.load(study)
                data = study_data['data']
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


if __name__ == "__main__":
    load_data(r"C:\Data\processed")
