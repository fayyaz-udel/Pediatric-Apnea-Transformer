import gc

import keras
from keras.callbacks import EarlyStopping

from metrics import Result, NMSE
from missing_modality.modality import *
from missing_modality.model import create_unimodal_model, create_multimodal_model

PHASE = "multimodal"  # unimodal, multimodal
DATA_PATH = "/home/hamedcan/d/nch_30x64_"
EPOCHS = 100
BATCH_SIZE = 256
MODALS = ["eog", "eeg", "resp", "spo2", "ecg", "co2"]
NOISE_RATIO = 0
MISS_RATIO = 0

FOLDS = 1
TRAIN = False

result = Result()
### DATASET ###
for fold in range(FOLDS):
    m_list = generate_modalities(MODALS)
    #####################################################################
    first = True
    for i in range(5):
        print('f' + str(i))
        data = np.load(DATA_PATH + str(i) + ".npz", allow_pickle=True)
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
    load_data(m_list, x_train, x_test, MISS_RATIO, NOISE_RATIO)
    gc.collect()
    keras.backend.clear_session()
    early_stopper = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    if PHASE == "unimodal":
        if TRAIN:
            model = create_unimodal_model(m_list)
            model.compile(optimizer='adam', loss=generate_loss(m_list), metrics='acc')
            history = model.fit(x=get_x_train(m_list), y=get_x_train(m_list) + [y_train] * len(m_list),
                                validation_split=0.1, epochs=EPOCHS, batch_size=BATCH_SIZE,
                                callbacks=[early_stopper])
            model.save_weights('./weights/uniweights_' + str(fold) + '.h5')

    elif PHASE == "multimodal":
        model = create_multimodal_model(m_list)
        if TRAIN:
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
            model.load_weights('./weights/uniweights_' + str(fold) + '.h5', by_name=True, skip_mismatch=True)
            history = model.fit(x=get_x_train(m_list), y=y_train, validation_split=0.1,
                                epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[early_stopper])
            model.save_weights('./weights/mulweights_f' + str(fold) + '.h5')
        else:
            model.load_weights('./weights/mulweights_f' + str(fold) + '.h5')
            predict = model.predict(get_x_test(m_list))
            y_predict = np.where(predict > 0.5, 1, 0)
            result.add(y_test, y_predict, predict)



    else:
        raise Exception("Invalid phase: " + PHASE)

if not TRAIN:
    result.print()
    result.save("./result/" + "test" + ".txt", str(MODALS) + str(DEL_MODALS) + str(NOISE_MODALS))
