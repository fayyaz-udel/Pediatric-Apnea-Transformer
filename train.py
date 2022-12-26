import keras
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.utils import Sequence
from sklearn.utils import shuffle
import networks
from metrics import Precision, Recall

DATA_PATH = "C:\\Data\\filtered_bmi_age_3threshold.npz"
THRESHOLD = 1
FOLD = 5


class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch - 5) % 10 == 0:
        lr *= 0.25
    print("Learning rate: ", lr)
    return lr


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="../model.h5",
        save_weights_only=True,
        save_best_only=True)


if __name__ == "__main__":
    data = np.load(DATA_PATH, allow_pickle=True)
    x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']
    y = y_apnea + y_hypopnea

    for i in range(FOLD):
        x[i], y[i] = shuffle(x[i], y[i])
        y[i] = np.where(y[i] >= THRESHOLD, 1, 0)
        # x[i] = x[i][:, :, :4] # CHANNEL SELECTION

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
        # train_gen = DataGenerator(x_train, y_train, 256)

        model = networks.create_transformer_model(input_shape=(90, 6),
                                                  num_patches=30, patch_size=3, projection_dim=32,
                                                  transformer_layers=8, num_heads=4, transformer_units=[64, 32],
                                                  mlp_head_units=[256, 128], num_classes=1)

        print(model.summary())

        opt = keras.optimizers.Adam()
        model.compile(optimizer=opt, loss=keras.losses.BinaryFocalCrossentropy(from_logits=True), metrics=['accuracy', Precision(from_logits=True), Recall(from_logits=True)])

        lr_scheduler = LearningRateScheduler(lr_schedule)
        early_stopper = EarlyStopping(patience=30, restore_best_weights=True)
        history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=100, validation_split=0.1,
                           callbacks=[early_stopper, lr_scheduler], class_weight={0: 1.0, 1: 2.0})

        model.save("./weights36432/fold " + str(fold))
