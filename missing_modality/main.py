import numpy as np

from missing_modality.modality import *
from missing_modality.model import *

PHASE = "unimodal"  # unimodal, multimodal
DATA_PATH = "/media/hamed/NSSR Dataset/nch_30x64_test_"
EPOCHS = 50

### DATASET ###
x, y = [], []
for i in range(5):
    data = np.load(DATA_PATH + str(i) + ".npz", allow_pickle=True)
    x.append(data['x'])
    y.append(np.sign(data['y_apnea'] + data['y_hypopnea']))

for fold in range(5):
    m_list = generate_modalities()
    keras.backend.clear_session()
    #####################################################################
    first = True
    for i in range(5):
        if i != fold:
            if first:
                x_train = x[i]
                y_train = y[i]
                first = False
            else:
                x_train = np.concatenate((x_train, x[i]))
                y_train = np.concatenate((y_train, y[i]))

    x_test = x[fold]
    y_test = y[fold]
    ######################################################################
    load_data(m_list, x_train, x_test)

    if PHASE == "unimodal":
        model = create_unimodal_model(m_list)
        model.compile(optimizer='adam', loss=generate_loss(m_list), metrics='acc')

        history = model.fit(x=get_x_train(m_list), y=get_x_train(m_list) + [y_train] * len(m_list), epochs=EPOCHS)
        print(model.evaluate(x=get_x_test(m_list), y=get_x_test(m_list) + [y_test] * len(m_list)))
        model.save_weights('./weights/uniweights.h5')




    elif PHASE == "multimodal":
        model = create_multimodal_model(m_list)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics='acc')
        model.load_weights('./weights/uniweights.h5', by_name=True, skip_mismatch=True)
        history = model.fit(x=get_x_train(m_list), y=y_train, validation_data= (get_x_test(m_list), y_test),epochs=EPOCHS)
        print(model.evaluate(x=get_x_test(m_list), y=y_test))
        model.save_weights('./weights/mulweights.h5')


    else:
        raise Exception("Invalid phase: " + PHASE)
