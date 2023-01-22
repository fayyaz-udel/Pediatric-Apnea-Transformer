import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


data = np.load(r"D:\Data\p600\10054_21487_303_139.npz", allow_pickle=True)

# y = y_apnea + y_hypopnea
# for i in range(5):
#     x[i] = np.nan_to_num(x[i], nan=-1)
#     y[i] = np.where(y[i] >= 1, 1, 0)
#     print(np.sum(y[i])/len(y[i]))

# print(signal.shape)
# for i in range(6):
#     #plt.plot(range(180), signal[1000,:180,i])
#     d = signal[:,:,i]
#     print(np.max(d), np.percentile(d,99.9), np.min(d), np.percentile(d,0.1))
plt.plot(data[0][100,1,1])
plt.show()
