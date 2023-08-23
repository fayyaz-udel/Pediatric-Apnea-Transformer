import keras
from keras import layers

from missing_modality.modality import get_inps, get_decs, get_clss, get_encs, get_a_s, get_f_enc_flats, get_f_a_s, \
    get_f_encs


def create_encoder_1D(modality_str, input_shape=(1920, 1), t=True):
    input = keras.Input(shape=input_shape)
    n = modality_str + "_enc"
    x = layers.Conv1D(16, 60, activation='relu', padding='same', name=n + '_l1', trainable=t)(input)
    x = layers.MaxPooling1D(5, padding='same')(x)
    x = layers.Conv1D(8, 60, activation='relu', padding='same', name=n + '_l2', trainable=t)(x)
    x = layers.MaxPooling1D(10, padding='same')(x)
    x = layers.Conv1D(8, 60, activation='relu', padding='same', name=n + '_l3', trainable=t)(x)
    encoded = layers.MaxPooling1D(10, padding='same')(x)

    return keras.Model(input, encoded, name=n)


def create_encoder_2D(modality_str, input_shape=(128, 16, 1), t=True):
    input = keras.Input(shape=input_shape)
    n = modality_str + "_enc"
    x = layers.Conv2D(16, (6, 3), activation='relu', padding='same', name=n + '_l1', trainable=t)(input)
    x = layers.MaxPooling2D((4, 2), padding='same')(x)
    x = layers.Conv2D(8, (6, 3), activation='relu', padding='same', name=n + '_l2', trainable=t)(x)
    x = layers.MaxPooling2D((4, 2), padding='same')(x)
    x = layers.Conv2D(8, (6, 3), activation='relu', padding='same', name=n + '_l3', trainable=t)(x)
    encoded = layers.MaxPooling2D((4, 2), padding='same')(x)
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    return keras.Model(input, encoded, name=n)


def create_encoder(modality_str, input_shape, t=True):
    if len(input_shape) == 3:
        return create_encoder_2D(modality_str, input_shape, t)
    elif len(input_shape) == 2:
        return create_encoder_1D(modality_str, input_shape, t)
    else:
        raise Exception("Invalid input shape: " + str(input_shape))


def create_decoder_1D(modality_str, input_shape=(4, 8), t=True):
    input = keras.Input(shape=input_shape)
    n = modality_str + "_dec"
    x = layers.Conv1D(12, 60, activation='relu', padding='same', name=n + '_l1', trainable=t)(input)
    x = layers.UpSampling1D(10)(x)
    x = layers.Conv1D(8, 60, activation='relu', padding='same', name=n + '_l2', trainable=t)(x)
    x = layers.UpSampling1D(8)(x)
    x = layers.Conv1D(5, 60, activation='relu', padding='same', name=n + '_l3', trainable=t)(x)
    x = layers.UpSampling1D(6)(x)
    decoded = layers.Conv1D(1, 6, activation='sigmoid', padding='same', name=n + '_l4', trainable=t)(x)

    return keras.Model(input, decoded, name=n)


def create_decoder_2D(modality_str, input_shape=(2, 2, 8), t=True):
    input = keras.Input(shape=input_shape)
    n = modality_str + "_dec"
    x = layers.Conv2D(8, (6, 3), activation='relu', padding='same', name=n + '_l1', trainable=t)(input)
    x = layers.UpSampling2D((4, 2))(x)
    x = layers.Conv2D(8, (6, 3), activation='relu', padding='same', name=n + '_l2', trainable=t)(x)
    x = layers.UpSampling2D((4, 2))(x)
    x = layers.Conv2D(16, (6, 3), activation='relu', padding='same', name=n + '_l3', trainable=t)(x)
    x = layers.UpSampling2D((4, 2))(x)
    decoded = layers.Conv2D(1, (6, 3), activation='sigmoid', padding='same', name=n + '_l4', trainable=t)(x)

    return keras.Model(input, decoded, name=n)


def create_decoder(modality_str, input_shape, t=True):
    if len(input_shape) == 3:
        return create_decoder_2D(modality_str, input_shape, t)
    elif len(input_shape) == 2:
        return create_decoder_1D(modality_str, input_shape, t)
    else:
        raise Exception("Invalid input shape: " + str(input_shape))


def create_fusion_network(m_list, HIDDEN_STATE_DIM=8):
    input_shape_a_s = 1
    for m in m_list:
        m.f_enc = keras.Input(m.z_dim)
        m.f_enc_flat = layers.Flatten()(m.f_enc)
        m.f_a_s = keras.Input(input_shape_a_s)

    x = layers.Concatenate()(get_f_enc_flats(m_list) + get_f_a_s(m_list))

    # m1 = input[:, :, 0]
    # m2 = input[:, :, 1]
    #
    # h_1 = layers.Dense(tf.reshape(m1, [-1, 1]), HIDDEN_STATE_DIM, activation=tf.nn.tanh)
    # h_2 = layers.Dense(tf.reshape(m2, [-1, 1]), HIDDEN_STATE_DIM, activation=tf.nn.tanh)
    # z = layers.Dense(tf.stack([m1, m2], axis=1), HIDDEN_STATE_DIM, activation=tf.nn.sigmoid)
    # h = z * h_1 + (1 - z) * h_2
    # label = layers.Dense(1, activation='sigmoid')(h)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    label = layers.Dense(1, activation='sigmoid')(x)
    get_f_encs(m_list) + get_f_a_s(m_list)
    return keras.Model(get_f_encs(m_list) + get_f_a_s(m_list), label, name='fusion')


def create_classifier(modality_str, inp_dim):
    input = keras.Input(inp_dim)
    n = modality_str + '_cls'
    x = layers.Flatten()(input)
    x = layers.Dense(64, activation='relu', name=n + '_l1')(x)
    x = layers.Dense(16, activation='relu', name=n + '_l2')(x)
    label = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(input, label, name=n)


def create_unimodal_model(m_list):
    for m in m_list:
        m.inp = keras.Input(m.inp_dim, name=m.name + '_inp')
        m.enc = create_encoder(m.name, m.inp_dim)(m.inp)
        m.dec = create_decoder(m.name, m.z_dim)(m.enc)
        m.cls = create_classifier(m.name, m.z_dim)(m.enc)

    return keras.Model(get_inps(m_list), get_decs(m_list) + get_clss(m_list))


def create_multimodal_model(m_list):
    for m in m_list:
        m.inp = keras.Input(m.inp_dim, name=m.name + '_inp')
        m.enc = create_encoder(m.name, m.inp_dim)(m.inp)
        m.dec = create_decoder(m.name, m.z_dim)(m.enc)
        m.cls = create_classifier(m.name, m.z_dim)(m.enc)

        m.inp_flat = layers.Flatten()(m.inp)
        m.dec_flat = layers.Flatten()(m.dec)
        m.a_s = keras.losses.mean_squared_error(m.inp_flat, m.dec_flat)

    ### FUSION NETWORK ###
    label = create_fusion_network(m_list)(get_encs(m_list) + get_a_s(m_list))
    return keras.Model(get_inps(m_list), label, name='multimodal_model')
