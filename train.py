import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.utils import shuffle

import networks

DATA_PATH = "C:\\Data\\filtered_balanced.npz"
MODEL_PATH = "./weights_bal_cls/fold "
THRESHOLD = 1
FOLD = 5


def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch - 5) % 10 == 0:
        lr *= 0.5
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea
    ########################################################################################
    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])

        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        # y[i] = np.sqrt(y[i])
        # y[i][y[i] != 0] += 2
        # x[i] = x[i][:, :, :4] # CHANNEL SELECTION
    ########################################################################################
    for fold in range(FOLD):
        first = True
        for i in range(FOLD):
            if i != fold:
                if first:
                    x_train = np.nan_to_num(x[i], nan=-1)
                    y_train = y[i]
                    first = False
                else:
                    x_train = np.concatenate((x_train, np.nan_to_num(x[i], nan=-1)))
                    y_train = np.concatenate((y_train, y[i]))

        # y_train = keras.utils.to_categorical(y_train, num_classes=2) # For MultiClass
        # y_test = keras.utils.to_categorical(y_test, num_classes=2) # For MultiClass
        ################################################################################################################
        model = networks.create_transformer_model(input_shape=(180, 6),
                                                  num_patches=60, patch_size=3, projection_dim=16,
                                                  transformer_layers=4, num_heads=4, transformer_units=[32, 16],
                                                  mlp_head_units=[256, 128], num_classes=1)

        model.compile(optimizer='adam', loss="mean_squared_error", metrics=["mean_absolute_error"])
        # 'accuracy', Precision(from_logits=True), Recall(from_logits=True) TODO

        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopper = EarlyStopping(patience=30, restore_best_weights=True)
        history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=100, validation_split=0.1,
                            callbacks=[early_stopper, lr_scheduler])
        # , class_weight={0: 0.5, 1: 2.0}

        model.save(MODEL_PATH + str(fold))
