import numpy as np

data_path = "D:\\Data\\chat_3_64.npz"
data = np.load(data_path, allow_pickle=True)
x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']

print("LOAD COMPLETE")
OUT_PATH = "D:\\Data\\chat_3_64_test.npz"
np.savez_compressed(OUT_PATH, x=x[1][:11195:100], y_apnea=y_apnea[1][:11195:100], y_hypopnea=y_hypopnea[1][:11195:100])