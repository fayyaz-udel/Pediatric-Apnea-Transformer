import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from metrics import Result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


THRESHOLD = 1
FOLD = 5


def test(config):
    result = Result()
    idx_list = np.load("idx1.npz")["idx_list"]
    data = np.load(config["data_path"], allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea

    first = True
    for i in range(FOLD):
        print(i)
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]
        ############################################################################
        if first:
            x_train = x[i]
            y_train = y[i]
            first = False
        else:
            x_train = np.concatenate((x_train, x[i]))
            y_train = np.concatenate((y_train, y[i]))
    ############################################################################
    min_age = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    max_age = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 99]
    for age_range in range(len(min_age)):
        # idx_list.append(((min_age[age_range] <= x_train[:,1,-1] * 24.99) & (max_age[age_range] > x_train[:,1,-1] * 24.99)))
        # print(sum(idx_list[age_range]))
        model = tf.keras.models.load_model(config["model_path"] + str(0), compile=False)

        predict = model.predict(x_train[idx_list[age_range], :, :])
        y_score = predict
        y_predict = np.where(predict > 0.5, 1, 0)  # For MultiClass np.argmax(y_score, axis=-1)

        result.add(y_train[idx_list[age_range]], y_predict, y_score)
    #np.savez_compressed("idx1", idx_list=idx_list)
    result.print()
    result.save("./results/" + config["model_name"] + str(age_range) + ".txt", config)
