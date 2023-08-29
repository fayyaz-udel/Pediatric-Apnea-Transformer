import numpy as np
from scipy import signal


class Modality:
    def __init__(self, name, index, inp_dim, z_dim, need_freq=False, need_reshape=False):
        self.name = name
        self.index = index
        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.dim = len(inp_dim) - 1
        self.inp = None
        self.inp_flat = None
        self.dec = None
        self.dec_flat = None
        self.enc = None
        self.cls = None
        self.a_s = None  # Anomaly Score
        ### fusion variables ###
        self.f_enc = None
        self.f_enc_flat = None
        self.f_a_s = None
        ### data ###
        self.x_train = None
        self.x_test = None
        self.need_freq = need_freq
        self.need_reshape = need_reshape


def get_inps(ms):
    inps = []
    for m in ms:
        inps.append(m.inp)
    return inps


def get_encs(ms):
    encs = []
    for m in ms:
        encs.append(m.enc)
    return encs


def get_decs(ms):
    decs = []
    for m in ms:
        decs.append(m.dec)
    return decs


def get_clss(ms):
    clss = []
    for m in ms:
        clss.append(m.cls)
    return clss


def get_a_s(ms):
    a_s = []
    for m in ms:
        a_s.append(m.a_s)
    return a_s


def get_f_encs(ms):
    f_enc = []
    for m in ms:
        f_enc.append(m.f_enc)
    return f_enc


def get_f_enc_flats(ms):
    f_enc_flat = []
    for m in ms:
        f_enc_flat.append(m.f_enc_flat)
    return f_enc_flat


def get_f_a_s(ms):
    f_a_s = []
    for m in ms:
        f_a_s.append(m.f_a_s)
    return f_a_s


def get_x_train(ms):
    x_train = []
    for m in ms:
        x_train.append(m.x_train)
    return x_train


def get_x_test(ms):
    x_test = []
    for m in ms:
        x_test.append(m.x_test)
    return x_test


def generate_loss(m_list, dec_loss='mae', cls_loss='binary_crossentropy'):
    loss = {}
    for m in m_list:
        loss[m.name + '_dec'] = dec_loss
        loss[m.name + '_cls'] = cls_loss
    return loss

def generate_loss_weights(m_list):
    loss_weights = {}
    for m in m_list:
        loss_weights[m.name + '_dec'] = 1
        loss_weights[m.name + '_cls'] = 1
    return loss_weights


def load_data(m_list, x_train, x_test, miss_modal=[], noise_modal={}):
    for m in m_list:
        ################  missing modality  #######################
        if m.name in miss_modal:
            x_train[:, :, m.index] = np.zeros_like(x_train[:, :, m.index])
            x_test[:, :, m.index] = np.zeros_like(x_test[:, :, m.index])
        ################  noisy modality   ########################
        elif m.name in list(noise_modal.keys()):
            x_train[:, :, m.index] = add_noise_to_data(x_train[:, :, m.index], target_snr_db=noise_modal[m.name])
            x_test[:, :, m.index] = add_noise_to_data(x_test[:, :, m.index], target_snr_db=noise_modal[m.name])
        ###########################################################
        if m.need_freq:
            m.x_train = transform2freq(x_train, m.index)
            m.x_test = transform2freq(x_test, m.index)
        elif m.need_reshape:
            m.x_train = resize(x_train, m.index)
            m.x_test = resize(x_test, m.index)
        else:
            m.x_train = x_train[:, :, m.index]
            m.x_test = x_test[:, :, m.index]
        ###########################################################
        ###########################################################
        m.x_train = normalize(m.x_train)
        m.x_test = normalize(m.x_test)


def normalize(xx):
    for i in range(xx.shape[-1]):
        x = xx[:, :, :, i]
        x = np.clip(x, np.percentile(x, 0.1), np.percentile(x, 99.9))
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        xx[:, :, :, i] = x
    return xx


############################################## NOISE/MISSING MODALITY ##################################################
def add_noise_to_signal(signal, target_snr_db=20):
    signal_watts = signal ** 2
    # Calculate signal power and convert to dB
    sig_avg_watts = np.mean(signal_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    y_noise = np.random.normal(0, np.sqrt(noise_avg_watts), (len(signal_watts), 1))
    return signal + y_noise


def add_noise_to_data(data, target_snr_db=20):
    for sample in range(data.shape[0]):
        data[sample, :, :] = add_noise_to_signal(data[sample, :, :], target_snr_db)
    return data


########################################################################################################################

# def get_augmentation_model():
#     model = keras.Sequential(
#         [layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),],
#         name="data_augmentation",
#     )
#     return model


def transform2freq(x, idx):
    out_x = np.zeros((x.shape[0], 128, 16, 1))
    for i in range(x.shape[0]):
        f, t, Zxx = signal.stft(x[i, :, idx], fs=64, padded=False)
        Zxx = np.squeeze(Zxx)
        Zxx = np.abs(Zxx)[:128, :16]
        out_x[i, :, :, 0] = ((Zxx - np.min(Zxx)) / (np.max(Zxx) - np.min(Zxx)))
    return np.nan_to_num(out_x)


def resize(x, idx):
    out_x = np.zeros((x.shape[0], 128, 16, len(idx)))
    for n, id in enumerate(idx):
        for i in range(x.shape[0]):
            out_x[i, :, :, n] = np.reshape(np.pad(x[i, :, id], [(0, 128)]), out_x.shape[1:3])
    return np.nan_to_num(out_x)


def generate_modalities(m_names):
    m_list = []
    modals = {
        "eog": Modality("eog", [0], (128, 16, 1), (16, 16, 1), need_freq=True),
        "eeg": Modality("eeg", [1], (128, 16, 1), (16, 16, 1), need_freq=True),
        "resp": Modality("resp", [2, 3], (128, 16, 2), (16, 16, 1), need_reshape=True),

        "spo2": Modality("spo2", [4], (128, 16, 1), (16, 16, 1), need_reshape=True),
        "co2": Modality("co2", [5], (128, 16, 1), (16, 16, 1), need_reshape=True),
        "ecg": Modality("ecg", [7, 8], (128, 16, 2), (16, 16, 1), need_reshape=True),

    }
    for m in m_names:
        m_list.append(modals[m])
    return m_list
