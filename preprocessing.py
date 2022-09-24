import concurrent.futures
from datetime import datetime

import sleep_study as ss
import numpy as np
import mne
import random


NUM_WORKER = 6
SN = 3984 # STUDY NUMBER
FREQ = 100
CHUNK_DURATION = 30.0
OUT_FOLDER = r'C:\Data\preprocessed'
channels = [
    'ECG EKG2-EKG',
]

POS_EVENT_DICT = {
    "Obstructive Apnea": 1,
    "Central Apnea": 1,
    "Mixed Apnea": 1,
    "apnea": 1,
    "obstructive apnea": 1,
    "central apnea": 1,
    "apnea": 1,
    "Apnea": 1,
}

NEG_EVENT_DICT = {
    'Sleep stage N1': 0,
    'Sleep stage N2': 0,
    'Sleep stage N3': 0,
    'Sleep stage R': 0,
}


mne.set_log_file('log.txt', overwrite=False)


def apnea2bad(df):
    df = df.replace(r'.*pnea.*', 'badevent', regex=True)
    print("bad replaced!")
    return df


def change_duration(df, label_dict=POS_EVENT_DICT, duration=CHUNK_DURATION):
    for key in label_dict:
        df.loc[df.description == key, 'duration'] = duration
    print("change duration!")
    return df


def preprocess(i, annotation_modifier, EVENT_DICT, out_dir, postfix, rnd):
    print(str(i) + "---" + str(datetime.now().time().strftime("%H:%M:%S")) + ' --- Processing %d' % i)
    study = ss.data.study_list[i]
    raw = ss.data.load_study(study, annotation_modifier, verbose=True)
    if not all([name in raw.ch_names for name in channels]):
        print("study " + str(study) + " skipped since insufficient channels")
        return 0
    try:
        events, event_ids = mne.events_from_annotations(raw, event_id=EVENT_DICT, chunk_duration=CHUNK_DURATION,
                                                        verbose=None)
    except ValueError:
        print("No Chunk found!")
        return 0
    sfreq = raw.info['sfreq']
    tmax = CHUNK_DURATION - 1. / sfreq
    epochs = mne.Epochs(raw=raw, picks=channels, events=events, event_id=event_ids, tmin=0., tmax=tmax,
                        baseline=None, verbose=None)
    epochs.load_data()
    epochs = epochs.resample(FREQ, npad='auto')
    data = epochs.get_data()
    labels = events[:, -1]

    c = 0
    for i in range(data.shape[0]):
        if random.randint(0, rnd) == 0:
            np.savez_compressed(out_dir + '\\' + study + postfix + str(i), data=data[i], labels=labels[i])
            c += 1
    return c


if __name__ == "__main__":
    ss.__init__()
    if NUM_WORKER < 2:
        for idx in range(SN):
            slots_number = preprocess(idx, change_duration, POS_EVENT_DICT, OUT_FOLDER, "abnorm", 0)
        for idx in range(SN):
            slots_number = preprocess(idx, apnea2bad, NEG_EVENT_DICT, OUT_FOLDER, "norm", 100)

    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKER) as executor:
            executor.map(preprocess, range(SN), [change_duration] * SN, [POS_EVENT_DICT] * SN, [OUT_FOLDER] * SN, ["abnorm"] * SN, [0] * SN)
            executor.map(preprocess, range(SN), [apnea2bad] * SN, [NEG_EVENT_DICT] * SN, [OUT_FOLDER] * SN, ["norm"] * SN, [100] * SN)
