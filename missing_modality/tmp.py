import numpy as np

DATA_PATH = "/media/hamed/NSSR Dataset/nch_30x64_"
x, y = [], []
for i in range(5):
    data = np.load(DATA_PATH + str(i) + ".npz", allow_pickle=True)
    x.append(data['x'])
    y.append(np.sign(data['y_apnea'] + data['y_hypopnea']))
    print(i)