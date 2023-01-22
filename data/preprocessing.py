import concurrent.futures
from datetime import datetime
import pandas as pd
import mne
import numpy as np
import scipy
import matplotlib
from numpy import array

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from biosppy.signals.ecg import ecg, hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from mne import make_fixed_length_events
from scipy.interpolate import splev, splrep
from itertools import compress
import sleep_study as ss
from ecgdetectors import Detectors

THRESHOLD = 3
NUM_WORKER = 8
SN = 3984  # STUDY NUMBER
FREQ = 256.0
CHANNELS_NO = 4
CHUNK_DURATION = 60.0
detectors = Detectors(FREQ)
OUT_FOLDER = 'C:\\Data\\p60'
channels = [
    'ECG EKG2-EKG',
    'RESP PTAF',
    'SPO2',
]

APNEA_EVENT_DICT = {
    "Obstructive Apnea": 2,
    "Central Apnea": 2,
    "Mixed Apnea": 2,
    "apnea": 2,
    "obstructive apnea": 2,
    "central apnea": 2,
    "apnea": 2,
    "Apnea": 2,
}

HYPOPNEA_EVENT_DICT = {
    "Obstructive Hypopnea": 1,
    "Hypopnea": 1,
    "hypopnea": 1,
    "Mixed Hypopnea": 1,
    "Central Hypopnea": 1,
}

POS_EVENT_DICT = {
    "Obstructive Hypopnea": 1,
    "Hypopnea": 1,
    "hypopnea": 1,
    "Mixed Hypopnea": 1,
    "Central Hypopnea": 1,

    "Obstructive Apnea": 2,
    "Central Apnea": 2,
    "Mixed Apnea": 2,
    "apnea": 2,
    "obstructive apnea": 2,
    "central apnea": 2,
    "Apnea": 2,
}

NEG_EVENT_DICT = {
    'Sleep stage N1': 0,
    'Sleep stage N2': 0,
    'Sleep stage N3': 0,
    'Sleep stage R': 0,
}

WAKE_DICT = {
    "Sleep stage W": 10
}

mne.set_log_file('log.txt', overwrite=False)


def identity(df):
    return df


def apnea2bad(df):
    df = df.replace(r'.*pnea.*', 'badevent', regex=True)
    print("bad replaced!")
    return df


def wake2bad(df):
    return df.replace("Sleep stage W", 'badevent')


def change_duration(df, label_dict=POS_EVENT_DICT, duration=CHUNK_DURATION):
    for key in label_dict:
        df.loc[df.description == key, 'duration'] = duration
    print("change duration!")
    return df


def preprocess(i, annotation_modifier, out_dir, ahi_dict):
    is_apnea_available, is_hypopnea_available = True, True
    study = ss.data.study_list[i]
    raw = ss.data.load_study(study, annotation_modifier, verbose=True)
    ########################################   CHECK CRITERIA FOR SS   #################################################
    if not all([name in raw.ch_names for name in channels]):
        print("study " + str(study) + " skipped since insufficient channels")
        return 0

    if ahi_dict.get(study, 0) < THRESHOLD:
        print("study " + str(study) + " skipped since low AHI ---  AHI = " + str(ahi_dict.get(study, 0)))
        return 0

    try:
        apnea_events, event_ids = mne.events_from_annotations(raw, event_id=POS_EVENT_DICT, chunk_duration=1.0,
                                                              verbose=None)
    except ValueError:
        print("No Chunk found!")
        return 0
    ########################################   CHECK CRITERIA FOR SS   #################################################
    print(str(i) + "---" + str(datetime.now().time().strftime("%H:%M:%S")) + ' --- Processing %d' % i)

    try:
        apnea_events, event_ids = mne.events_from_annotations(raw, event_id=APNEA_EVENT_DICT, chunk_duration=1.0,
                                                              verbose=None)
    except ValueError:
        is_apnea_available = False

    try:
        hypopnea_events, event_ids = mne.events_from_annotations(raw, event_id=HYPOPNEA_EVENT_DICT, chunk_duration=1.0,
                                                                 verbose=None)
    except ValueError:
        is_hypopnea_available = False

    wake_events, event_ids = mne.events_from_annotations(raw, event_id=WAKE_DICT, chunk_duration=1.0, verbose=None)
    ####################################################################################################################
    sfreq = raw.info['sfreq']
    tmax = CHUNK_DURATION - 1. / sfreq

    fixed_events = make_fixed_length_events(raw, id=0, duration=CHUNK_DURATION, overlap=0.)
    epochs = mne.Epochs(raw, fixed_events, event_id=[0], tmin=0, tmax=tmax, baseline=None, preload=True, proj=False,
                         verbose=None)
    # picks=channels,
    epochs.load_data()
    if sfreq != FREQ:
        epochs = epochs.resample(FREQ, npad='auto', n_jobs=4, verbose=None)
    data = epochs.get_data()
    ####################################################################################################################
    if is_apnea_available:
        apnea_events_set = set((apnea_events[:, 0] / sfreq).astype(int))
    if is_hypopnea_available:
        hypopnea_events_set = set((hypopnea_events[:, 0] / sfreq).astype(int))
    wake_events_set = set((wake_events[:, 0] / sfreq).astype(int))

    starts = (epochs.events[:, 0] / sfreq).astype(int)

    labels_apnea = []
    labels_hypopnea = []
    labels_wake = []
    total_apnea_event_second = 0
    total_hypopnea_event_second = 0

    for seq in range(data.shape[0]):
        epoch_set = set(range(starts[seq], starts[seq] + 60))
        if is_apnea_available:
            apnea_seconds = len(apnea_events_set.intersection(epoch_set))
            total_apnea_event_second += apnea_seconds
            labels_apnea.append(apnea_seconds)
        else:
            labels_apnea.append(0)

        if is_hypopnea_available:
            hypopnea_seconds = len(hypopnea_events_set.intersection(epoch_set))
            total_hypopnea_event_second += hypopnea_seconds
            labels_hypopnea.append(hypopnea_seconds)
        else:
            labels_hypopnea.append(0)

        labels_wake.append(len(wake_events_set.intersection(epoch_set)) == 0)
    ####################################################################################################################
    print(study + "    HAMED    " + str(len(labels_wake) - sum(labels_wake)))
    data = data[labels_wake, :, :]
    labels_apnea = list(compress(labels_apnea, labels_wake))
    labels_hypopnea = list(compress(labels_hypopnea, labels_wake))

    data, idxs = process_ECG(data)
    data = data[idxs, :, :]
    labels_apnea = list(array(labels_apnea)[idxs])
    labels_hypopnea = list(array(labels_hypopnea)[idxs])

    if len(idxs) > 10:
        np.savez_compressed(
            out_dir + '\\' + study + "_" + str(total_apnea_event_second) + "_" + str(total_hypopnea_event_second),
            data=data, labels_apnea=labels_apnea, labels_hypopnea=labels_hypopnea)

    return data.shape[0]


def process_ECG(data):
    idx = []
    sleep_epoch_number = data.shape[0]
    ir = 3  # INTERPOLATION RATE(3HZ)
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    X = np.zeros((sleep_epoch_number, 180, CHANNELS_NO))

    for i in range(sleep_epoch_number):
        signal = np.squeeze(data[i, 0])
        filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                          frequency=[3, 45], sampling_rate=FREQ, )
        (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
        (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)
        # plt.clf()
        # plt.plot(rpeaks, signal[rpeaks], marker="o")
        # plt.plot(signal)

        if 40 < len(rpeaks) < 120 and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
            # plt.show()
            idx.append(i)
            rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
            ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
            rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
            amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)
            X[i, :, 0] = rri_interp_signal
            X[i, :, 1] = amp_interp_signal

        X[i, :, 2] = scipy.signal.resample(data[i, 1], int(CHUNK_DURATION * 3))
        X[i, :, 3] = scipy.signal.resample(data[i, 2], int(CHUNK_DURATION * 3))
    return X, idx


if __name__ == "__main__":
    ahi = pd.read_csv(r"C:\Data\AHI.csv")
    ahi_dict = dict(zip(ahi.Study, ahi.AHI))
    ss.__init__()

    if NUM_WORKER < 2:
        for idx in range(SN):
            preprocess(idx, identity, OUT_FOLDER, ahi_dict)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKER) as executor:
            executor.map(preprocess, range(SN), [identity] * SN, [OUT_FOLDER] * SN, [ahi_dict] * SN)
