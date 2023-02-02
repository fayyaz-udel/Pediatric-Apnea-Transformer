import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from biosppy.signals.ecg import ecg, hamilton_segmenter, correct_rpeaks
from biosppy.signals import tools as st
from mne import make_fixed_length_events
from scipy.interpolate import splev, splrep
from itertools import compress
import sleep_study as ss
from ecgdetectors import Detectors
from scipy.signal import resample
from sklearn.preprocessing import normalize


def extract_rri(signal, ir, CHUNK_DURATION):
    tm = np.arange(0, CHUNK_DURATION, step=1 / float(ir))  # TIME METRIC FOR INTERPOLATION

    filtered, _, _ = st.filter_signal(signal=signal, ftype="FIR", band="bandpass", order=int(0.3 * FREQ),
                                      frequency=[3, 45], sampling_rate=FREQ, )
    (rpeaks,) = hamilton_segmenter(signal=filtered, sampling_rate=FREQ)
    (rpeaks,) = correct_rpeaks(signal=filtered, rpeaks=rpeaks, sampling_rate=FREQ, tol=0.05)

    if 4 < len(rpeaks) < 200: # and np.max(signal) < 0.0015 and np.min(signal) > -0.0015:
        rri_tm, rri_signal = rpeaks[1:] / float(FREQ), np.diff(rpeaks) / float(FREQ)
        ampl_tm, ampl_signal = rpeaks / float(FREQ), signal[rpeaks]
        rri_interp_signal = splev(tm, splrep(rri_tm, rri_signal, k=3), ext=1)
        amp_interp_signal = splev(tm, splrep(ampl_tm, ampl_signal, k=3), ext=1)

        return np.clip(rri_interp_signal, 0, 2), np.clip(amp_interp_signal, -0.001, 0.002)
    else:
        return np.zeros((32 * 60)), np.zeros((32 * 60))


FREQ = 256
CHUNK_DURATION = 60.0
im = np.load(r"C:\Data\raw_allscipyECG.npz", allow_pickle=True)
data = im['x']
out = []
for f in range(5):
    sh = (data[f]).shape
    output = np.zeros((sh[0], 60 * 32, 2))
    for s in range(sh[0]):
        print(str(f) + "---" + str(s) + "  out of" + str(sh[0]))
        output[s, :, 0], output[s, :, 1] = extract_rri(data[f][s, :, 0],32, 60.0)

    out.append(output)

np.savez_compressed("C:\\Data\\raw_allscipyRRRHAMED", x=out, y_apnea=im["y_apnea"], y_hypopnea=im["y_hypopnea"])


# im = np.load(r"C:\Data\raw_allscipyRRR.npz", allow_pickle=True)
# data = im['x']
# for i in range(5):
#     data[i][:, :, 0] = np.clip(data[i][:, :, 0], 0, 2)
#     data[i][:, :, 1] = np.clip(data[i][:, :, 1], -0.001, 0.002)
#
# np.savez_compressed("C:\\Data\\raw_allscipyRRRR", x=data, y_apnea=im["y_apnea"], y_hypopnea=im["y_hypopnea"])
#
