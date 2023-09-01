import keras
from keras import layers
import tensorflow as tf
from missing_modality.modality import get_inps, get_decs, get_clss, get_encs, get_a_s, get_f_enc_flats, get_f_a_s, \
    get_f_encs, generate_modalities
from missing_modality.model_2d import create_decoder_2d, create_encoder_2d


def create_fusion_network(m_list, HIDDEN_STATE_DIM=16):
    input_shape_a_s = 1
    for m in m_list:
        m.f_enc = keras.Input(m.z_dim, name=m.name + '_f_z_inp')
        m.f_enc_flat = layers.Flatten(name=m.name + '_f_enc_flat')(m.f_enc)
        m.f_a_s = keras.Input(input_shape_a_s,name=m.name + '_f_q_inp')
        m.f_hd = layers.Dense(HIDDEN_STATE_DIM, activation=tf.nn.tanh, name=m.name + '_f_h')(m.f_enc_flat)
        m.f_h = layers.Dropout(rate=0.05)(m.f_hd)

    First = True
    z_stack = tf.concat(get_f_enc_flats(m_list), 1, name='z_stack')
    for m in m_list:
        m.f_zd = layers.Dense(HIDDEN_STATE_DIM, activation='sigmoid', name=m.name + "_z")(z_stack)
        m.f_z = layers.Dropout(rate=0.05)(m.f_zd)
        m.f_zh = layers.Multiply(name=m.name + "_zh")([m.f_z, m.f_h])

    for m in m_list:
        if First:
            h = m.f_zh
            First = False
        else:
            h = layers.Add(name=m.name + "_add")([h, m.f_zh])
    h_flat = layers.Flatten()(h)
    label = layers.Dense(1, activation='sigmoid')(h_flat)
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
        m.enc = create_encoder_2d(m.name, m.inp_dim)(m.inp)
        m.dec = create_decoder_2d(m.name, m.z_dim, output_shape=m.inp_dim)(m.enc)
        m.cls = create_classifier(m.name, m.z_dim)(m.enc)

    return keras.Model(get_inps(m_list), get_decs(m_list) + get_clss(m_list))


def create_multimodal_model(m_list):
    for m in m_list:
        m.inp = keras.Input(m.inp_dim, name=m.name + '_inp')
        m.enc = create_encoder_2d(m.name, m.inp_dim, trainable=False)(m.inp)
        m.dec = create_decoder_2d(m.name, m.z_dim, output_shape=m.inp_dim, trainable=False)(m.enc)

        m.inp_flat = layers.Flatten()(m.inp)
        m.dec_flat = layers.Flatten()(m.dec)
        m.a_s = keras.losses.mean_squared_error(m.inp_flat, m.dec_flat)

    ### FUSION NETWORK ###
    label = create_fusion_network(m_list)(get_encs(m_list) + get_a_s(m_list))
    return keras.Model(get_inps(m_list), label, name='multimodal_model')


if __name__ == "__main__":
    MODALS = ["eog", "eeg", "resp", "spo2", "ecg", "co2"]

    m_list = generate_modalities(MODALS)
    model = create_fusion_network(m_list)
    model.summary()
    keras.utils.plot_model(model, 'model.png', show_shapes=True)

