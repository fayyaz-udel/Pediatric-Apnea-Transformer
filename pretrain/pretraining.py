import keras
import numpy as np
from keras.callbacks import LearningRateScheduler, EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from data_generator import get_all_data
from models import create_cnn_model, create_vit_classifier, create_ed_cnn_model


def lr_schedule(epoch, lr):
    if epoch > 30 and (epoch - 1) % 5 == 0:
        lr *= 0.25
    print("Learning rate: ", lr)
    return lr


if __name__ == "__main__":
    X, y = get_all_data(r"C:\Data\preprocessed_filtered")
    y = keras.utils.to_categorical(y, num_classes=2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True)

for train, test in kfold.split(X, y.argmax(1)):
    model = create_ed_cnn_model()

    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='mae')

    lr_scheduler = LearningRateScheduler(lr_schedule)
    early_stopper = EarlyStopping(patience=20, restore_best_weights=True)
    history = model.fit(x=X[train], y=X[train], batch_size=256, epochs=10, validation_split=0.1, callbacks = [early_stopper, lr_scheduler])
    model.save_weights(r"..\weights\w")