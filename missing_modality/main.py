import gc
from datetime import datetime

import keras
from keras.callbacks import EarlyStopping

from metrics import Result
from missing_modality.modality import *
from missing_modality.model import create_unimodal_model, create_multimodal_model

config = {

    "PHASE": "multimodal",  # unimodal, multimodal
    # "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
    "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_test_",

    "EPOCHS": 100,
    "BATCH_SIZE": 256,
    "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
    "NOISE_RATIO": 0.00,
    "MISS_RATIO": 0.00,
    "NOISE_CHANCE": 0.0,
    "FOLDS": 1,
    "TRAIN": True,
}


def train(config):
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
        load_data(m_list, x_train, x_test, config["MISS_RATIO"], config["NOISE_RATIO"], config["NOISE_CHANCE"])
        gc.collect()
        keras.backend.clear_session()
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        if config["PHASE"] == "unimodal":
            if config["TRAIN"]:
                model = create_unimodal_model(m_list)
                model.compile(optimizer='adam', loss=generate_loss(m_list), metrics='acc')
                history = model.fit(x=get_x_train(m_list), y=get_x_train(m_list) + [y_train] * len(m_list),
                                    validation_split=0.1, epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
                                    callbacks=[early_stopper])
                model.save_weights('./weights/uniweights_' + str(fold) + '.h5')

        elif config["PHASE"] == "multimodal":
            model = create_multimodal_model(m_list)
            if config["TRAIN"]:
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
                # model.load_weights('./weights/uniweights_' + str(fold) + '.h5', by_name=True, skip_mismatch=True)
                history = model.fit(x=get_x_train(m_list), y=y_train, validation_split=0.1,
                                    epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"], callbacks=[early_stopper])
                model.save_weights('./weights/mulweights_f' + str(fold) + '.h5')
            else:
                model.load_weights('./weights/mulweights_f' + str(fold) + '.h5')
                predict = model.predict(get_x_test(m_list))
                y_predict = np.where(predict > 0.5, 1, 0)
                result.add(y_test, y_predict, predict)


        else:
            raise Exception("Invalid phase: " + config["PHASE"])

    if not config["TRAIN"]:
        result.print()
        result.save("./result/" + "test_" + config["MODEL_NAME"] + ".txt", config)


if __name__ == "__main__":
    train(config)
