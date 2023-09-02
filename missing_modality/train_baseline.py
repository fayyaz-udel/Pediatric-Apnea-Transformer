import gc
from datetime import datetime

import keras
from keras.callbacks import EarlyStopping
from keras.src.losses import BinaryCrossentropy

from metrics import Result
from missing_modality.modality import *
from missing_modality.model import create_unimodal_model, create_multimodal_model
from models.models import get_model

config = {
    "MODEL_NAME": "Transformer",
    "PHASE": "multimodal",  # unimodal, multimodal
    # "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
    "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_",

    "EPOCHS": 100,
    "BATCH_SIZE": 256,
    "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
    "NOISE_RATIO": 0.00,
    "MISS_RATIO": 0.00,
    "NOISE_CHANCE": 0.0,
    "FOLDS": 1,
    "TRAIN": True,
    ########################################################
    "transformer_layers": 5,  # best 5
    "drop_out_rate": 0.25,  # best 0.25
    "num_patches": 30,  # best 30 TBD
    "transformer_units": 32,  # best 32
    "regularization_weight": 0.001,  # best 0.001
    "num_heads": 4,
    "epochs": 100,  # best 200
    "channels": [0, 3, 5, 6, 9, 10, 4],
}


def  train_baseline(config):
    result = Result()
    ### DATASET ###
    for fold in range(config["FOLDS"]):
        m_list = generate_modalities(config["MODALS"])
        #####################################################################
        first = True
        for i in range(5):
            print('f' + str(i))
            data = np.load(config["DATA_PATH"] + str(i) + ".npz", allow_pickle=True)
            if i != fold:
                if first:
                    x_train = data['x']
                    y_train = np.sign(data['y_apnea'] + data['y_hypopnea'])
                    first = False
                else:
                    x_train = np.concatenate((x_train, data['x']))
                    y_train = np.concatenate((y_train, np.sign(data['y_apnea'] + data['y_hypopnea'])))
            else:
                x_test = data['x']
                y_test = np.sign(data['y_apnea'] + data['y_hypopnea'])
            del data
        ######################################################################
        ######################################################################
        x_train, x_test = load_data(m_list, x_train, x_test, config["MISS_RATIO"], config["NOISE_RATIO"], config["NOISE_CHANCE"], return_data=True)
        print(np.mean(x_test))
        gc.collect()
        keras.backend.clear_session()
        model = get_model(config)
        model.build((None, 1920, 7))
        model.compile(optimizer="adam", loss=BinaryCrossentropy(), metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

        if config["TRAIN"]:
            early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(x=x_train, y=y_train, batch_size=512, epochs=config["EPOCHS"], validation_split=0.1, callbacks=[early_stopper])
            model.save_weights('./weights/model_' + config["MODEL_NAME"] + "_" + str(fold) + '.h5')

        else:
            model.load_weights('./weights/model_' + config["MODEL_NAME"] + "_" + str(fold) + '.h5')
            predict = model.predict(x_test)
            y_predict = np.where(predict > 0.5, 1, 0)
            result.add(y_test, y_predict, predict)

    if not config["TRAIN"]:
        #result.print()
        print("==================")
        result.save("./result/" + "test_" + config["log_name"] + ".txt", config)


if __name__ == "__main__":
    train_baseline(config)
