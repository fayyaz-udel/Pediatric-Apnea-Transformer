import glob
import os
import random
import numpy as np
import pandas as pd
from scipy.signal import resample
from biosppy.signals.ecg import hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from scipy.interpolate import splev, splrep

# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1

# "EEG F3-M2",  # 2
# "EEG F4-M1",  # 3
# "EEG C3-M2",  # 4
# "EEG C4-M1",  # 5
# "EEG O1-M2",  # 6
# "EEG O2-M1",  # 7
# "EEG CZ-O1",  # 8

# "ECG EKG2-EKG",  # 9

# "RESP PTAF",  # 10
# "RESP AIRFLOW",  # 11
# "RESP THORACIC",  # 12
# "RESP ABDOMINAL",  # 13

# "SPO2",  # 14
# "RATE",  # 15
# "CAPNO",  # 16
# "RESP RATE",  # 17

# RRI
# Ramp
# Demo
# "EOG"
from data.tfrec import write_signals_to_tfr_short

SIGS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
s_count = len(SIGS)

THRESHOLD = 3
PATH = "D:\\data256s\\"
FREQ = 256
TARGET_FREQ = 32


def extract_rri(signal, ir, CHUNK_DURATION):
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                      frequency=[3, 45], sampling_rate=FREQ, )
    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
    (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)

    if 4 < len(rpeaks) < 200:  # and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
        rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
        ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)

        return np.clip(rri_interp_signal, 0, 2), np.clip(amp_interp_signal, -0.001, 0.002)
    else:
        return np.zeros((TARGET_FREQ * 60)), np.zeros((TARGET_FREQ * 60))


def load_data(path):
    demo = pd.read_csv("../misc/result.csv")
    ahi = pd.read_csv(r"C:\Data\AHI.csv")
    ahi_dict = dict(zip(ahi.Study, ahi.AHI))
    root_dir = os.path.expanduser(path)
    file_list = os.listdir(root_dir)
    length = len(file_list)

    study_event_counts = {}
    apnea_event_counts = {}
    hypopnea_event_counts = {}
    ######################################## Count the respiratory events ###########################################
    for i in range(length):
        patient_id = (file_list[i].split("_")[0])
        study_id = (file_list[i].split("_")[1])
        apnea_count = int((file_list[i].split("_")[2]))
        hypopnea_count = int((file_list[i].split("_")[3]).split(".")[0])

        if ahi_dict.get(patient_id + "_" + study_id, 0) > THRESHOLD:
            apnea_event_counts[patient_id] = apnea_event_counts.get(patient_id, 0) + apnea_count
            hypopnea_event_counts[patient_id] = hypopnea_event_counts.get(patient_id, 0) + hypopnea_count
            study_event_counts[patient_id] = study_event_counts.get(patient_id, 0) + apnea_count + hypopnea_count
        else:
            os.remove(PATH + file_list[i])

    apnea_event_counts = sorted(apnea_event_counts.items(), key=lambda item: item[1])
    hypopnea_event_counts = sorted(hypopnea_event_counts.items(), key=lambda item: item[1])
    study_event_counts = sorted(study_event_counts.items(), key=lambda item: item[1])

    ################################### Fold the data based on number of respiratory events #########################
    folds = []
    for i in range(5):
        folds.append(study_event_counts[i::5])

    x = []
    y_apnea = []
    y_hypopnea = []
    counter = 0
    for idx, fold in enumerate(folds):
        first = True
        for patient in fold:
            counter += 1
            print(counter)
            for study in glob.glob(PATH + patient[0] + "_*"):
                study_data = np.load(study)

                signals = study_data['data']
                labels_apnea = study_data['labels_apnea']
                labels_hypopnea = study_data['labels_hypopnea']

                identifier = study.split('\\')[-1].split('_')[0] + "_" + study.split('\\')[-1].split('_')[1]
                demo_arr = demo[demo['id'] == identifier].drop(columns=['id']).to_numpy().squeeze()

                y_c = labels_apnea + labels_hypopnea
                neg_samples = np.where(y_c == 0)[0]
                pos_samples = list(np.where(y_c > 0)[0])
                ratio = len(pos_samples) / len(neg_samples)
                neg_survived = []
                for s in range(len(neg_samples)):
                    if random.random() < ratio:
                        neg_survived.append(neg_samples[s])
                samples = neg_survived + pos_samples
                signals = signals[samples, :, :]
                labels_apnea = labels_apnea[samples]
                labels_hypopnea = labels_hypopnea[samples]

                data = np.zeros((signals.shape[0], 60 * TARGET_FREQ, s_count + 3))
                for i in range(signals.shape[0]):  # for each epoch
                    data[i, :len(demo_arr), -1] = demo_arr
                    data[i, :, -2], data[i, :, -3] = extract_rri(signals[i, 9, :], TARGET_FREQ, 60.0)
                    for j in range(s_count):  # for each signal
                        data[i, :, j] = resample(signals[i, SIGS[j], :], 60 * TARGET_FREQ)

                if first:
                    aggregated_data = data
                    aggregated_label_apnea = labels_apnea
                    aggregated_label_hypopnea = labels_hypopnea
                    first = False
                else:
                    aggregated_data = np.concatenate((aggregated_data, data), axis=0)
                    aggregated_label_apnea = np.concatenate((aggregated_label_apnea, labels_apnea), axis=0)
                    aggregated_label_hypopnea = np.concatenate((aggregated_label_hypopnea, labels_hypopnea), axis=0)

        write_signals_to_tfr_short(aggregated_data, aggregated_label_apnea, aggregated_label_hypopnea,
                                   filename="fold" + str(idx))
        # x.append(aggregated_data)
        # y_apnea.append(aggregated_label_apnea)
        # y_hypopnea.append(aggregated_label_hypopnea)

    return x, y_apnea, y_hypopnea

if __name__ == "__main__":
    x, y_apnea, y_hypopnea = load_data(PATH)
    # np.savez_compressed("C:\\Data\\raw_all_RR_demo", x=x, y_apnea=y_apnea, y_hypopnea=y_hypopnea)
