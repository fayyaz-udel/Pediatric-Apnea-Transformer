from sklearn.utils import shuffle
import tensorflow as tf
from metrics import Result

from sklearn.manifold import TSNE
import numpy as np
from  matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


THRESHOLD = 1
FOLD = 5


def test_tsne(config):
    data = np.load(config["data_path"], allow_pickle=True)
    ############################################################################
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        x[i] = np.nan_to_num(x[i], nan=-1)
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        x[i] = x[i][:, :, config["channels"]]
    ############################################################################

    for fold in [3]:
        x_test = x[4]
        y_test = y[4]  # For MultiClass keras.utils.to_categorical(y[fold], num_classes=2)

        model = tf.keras.models.load_model(config["model_path"] + str(fold), compile=False)
        model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
        # test_ds = np.concatenate(list(train_ds.take(5).map(lambda x, y: x)))  # get five batches of images and convert to numpy array
        features = model2(x_test)
        labels = y_test
        tsne = TSNE(n_components=2, learning_rate='auto',init='pca').fit_transform(features)
        tx = scale_to_01_range(tsne[:, 0])
        ty = scale_to_01_range(tsne[:, 1])

        colors = ['red', 'blue']
        classes = ['Apnea', 'Normal']
        print(classes)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for idx, c in enumerate(colors):
            indices = [i for i, l in enumerate(labels) if idx == l]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            ax.scatter(current_tx, current_ty, c=c, label=classes[idx])

        ax.legend(loc='best')
        plt.show()











