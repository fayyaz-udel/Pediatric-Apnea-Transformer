import keras
import numpy as np
from keras.callbacks import EarlyStopping
from keras.losses import BinaryCrossentropy

from apneaDetection_transformer.models.models import get_model
from metrics import Result
from missing_modality.models.modality import generate_modalities, load_data, generate_loss, get_x_train, get_x_test
from missing_modality.models.model import create_unimodal_model, create_multimodal_model

config = {
    "MODEL_NAME": "qaf",
    "STEP": "unimodal",  # unimodal, multimodal
    "DATA_PATH": "/home/hamedcan/dd/chat_b_30x64_",
    # "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_",
    "DATA_NAME": "chat",
    "EPOCHS": 100,
    "BATCH_SIZE": 256,
    "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
    "NOISE_RATIO": 0.00,
    "MISS_RATIO": 0.00,
    "NOISE_CHANCE": 0.0,
    "FOLDS": [0, 1, 2, 3, 4],
    "PHASE": ["TRAIN"],  # TRAIN, TEST
    ### Transformer Config  ######################
    "transformer_layers": 5,  # best 5
    "drop_out_rate": 0.25,  # best 0.25
    "num_patches": 30,  # best 30 TBD
    "transformer_units": 32,  # best 32
    "regularization_weight": 0.001,  # best 0.001
    "num_heads": 4,
    "epochs": 100,  # best 200
    "channels": [0, 3, 5, 6, 9, 10, 4],
}


def train_test(config):
    result = Result()
    ### DATASET ###
    for fold in config["FOLDS"]:
        m_list = generate_modalities(config["MODALS"])
        #####################################################################
        first = True
        for i in range(5):
            print('fold ' + str(i))
            data = np.load(config["DATA_PATH"] + str(i) + ".npz", allow_pickle=True)
            if i != fold and "TRAIN" in config["PHASE"]:
                if first:
                    x_train = data['x']
                    y_train = np.sign(data['y_apnea'] + data['y_hypopnea'])
                    first = False
                else:
                    x_train = np.concatenate((x_train, data['x']))
                    y_train = np.concatenate((y_train, np.sign(data['y_apnea'] + data['y_hypopnea'])))
            if i == fold and "TEST" in config["PHASE"]:
                x_test = data['x']
                y_test = np.sign(data['y_apnea'] + data['y_hypopnea'])
        ######################################################################

        if config["MODEL_NAME"] == 'qaf':
            load_data(m_list, x_train, x_test, config["MISS_RATIO"], config["NOISE_RATIO"], config["NOISE_CHANCE"],
                      config.get("MISS_INDEX", None))
        else:
            x_train, x_test = load_data(m_list, x_train, x_test, config["MISS_RATIO"], config["NOISE_RATIO"],
                                        config["NOISE_CHANCE"], config.get("MISS_INDEX", None), return_data=True)
        ######################################################################
        early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ### GET MODEL ###
        if config["MODEL_NAME"] == 'qaf':
            if config["STEP"] == 'unimodal':
                model = create_unimodal_model(m_list)
                model.compile(optimizer='adam', loss=generate_loss(m_list), metrics='acc')
            elif config["STEP"] == 'multimodal':
                model = create_multimodal_model(m_list)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
        else:
            model = get_model(config)
            model.build((None, 1920, 7))
            model.compile(optimizer="adam", loss=BinaryCrossentropy(),
                          metrics=[keras.metrics.Precision(), keras.metrics.Recall()])

        if config["STEP"] == "unimodal":
            if "TRAIN" in config["PHASE"]:
                if config["MODEL_NAME"] == 'qaf':
                    history = model.fit(x=get_x_train(m_list), y=get_x_train(m_list) + [y_train] * len(m_list),
                                        validation_split=0.1, epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
                                        callbacks=[early_stopper])
                    model.save_weights('./weights/uniweights_' + config["DATA_NAME"] + "_f" + str(fold) + '.h5')
                else:
                    model.fit(x=x_train, y=y_train, batch_size=config["BATCH_SIZE"], epochs=config["EPOCHS"],
                              validation_split=0.1, callbacks=[early_stopper])
                    model.save_weights(
                        './weights/model_' + config["MODEL_NAME"] + "_" + config["DATA_NAME"] + "_" + str(fold) + '.h5')
            if "TEST" in config["PHASE"]:
                if config["MODEL_NAME"] == 'qaf':
                    raise Exception("Unimodal test is not implemented yet for qaf")
                else:
                    model.load_weights(
                        './weights/model_' + config["MODEL_NAME"] + "_" + config["DATA_NAME"] + "_" + str(fold) + '.h5')
                    predict = model.predict(x_test)
                    y_predict = np.where(predict > 0.5, 1, 0)
                    result.add(y_test, y_predict, predict)

        elif config["STEP"] == "multimodal":
            if config["MODEL_NAME"] == 'qaf':
                if "TRAIN" in config["PHASE"]:
                    model.load_weights('./weights/uniweights_' + config["DATA_NAME"] + "_f" + str(fold) + '.h5',
                                       by_name=True, skip_mismatch=True)

                    for m in m_list:
                        m.dec.trainable = False
                        m.enc.trainable = False
                    history = model.fit(x=get_x_train(m_list), y=y_train, validation_split=0.1,
                                        epochs=config["EPOCHS"], batch_size=config["BATCH_SIZE"],
                                        callbacks=[early_stopper])
                    model.save_weights('./weights/mulweights_' + config["DATA_NAME"] + "_f" + str(fold) + '.h5')
                if "TEST" in config["PHASE"]:
                    model.load_weights('./weights/mulweights_' + config["DATA_NAME"] + "_f" + str(fold) + '.h5')
                    predict = model.predict(get_x_test(m_list))
                    y_predict = np.where(predict > 0.5, 1, 0)
                    result.add(y_test, y_predict, predict)
            else:
                raise Exception("Multimodal is just avaialble for QAF")

    if "TEST" in config["PHASE"]:
        result.print()
        result.save("./result/" + "test_" + config["MODEL_NAME"] + "_" + config["DATA_NAME"] + ".txt", config)


if __name__ == "__main__":
    train_test(config)
