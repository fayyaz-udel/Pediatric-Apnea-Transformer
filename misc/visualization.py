import os

import mne

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

raw = mne.io.edf.edf.RawEDF(input_fname=r"D:\nchsdb\sleep_data\46_11305.edf", preload=True,
                                verbose=True)

channels = [
    'ECG EKG2-EKG',
    'Resp PTAF',
    'SpO2',
]

raw = raw.pick_channels(channels)

raw.crop(tmin=4800, tmax=4805).load_data()

raw.plot()