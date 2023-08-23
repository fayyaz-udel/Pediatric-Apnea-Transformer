from missing_modality.util import transform2freq


class Modality:
    def __init__(self, name, index, inp_dim, z_dim, need_freq=False):
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


def load_data(m_list, x_train, x_test):
    for m in m_list:
        if m.need_freq:
            m.x_train = transform2freq(x_train, m.index)
            m.x_test = transform2freq(x_test, m.index)
        else:
            m.x_train = x_train[:, :, m.index]
            m.x_test = x_test[:, :, m.index]


def generate_modalities():
    m_list = []
    m_list.append(Modality("eog", [0], (128, 16, 1), (2, 2, 8), need_freq=True))
    m_list.append(Modality("eeg", [1], (128, 16, 1), (2, 2, 8), need_freq=True))
    # m_list.append(Modality("resp", [2], (1920, 1), (4, 8)))
    #
    # m_list.append(Modality("af", [3], (1920, 1), (4, 8)))
    # m_list.append(Modality("spo2", [4], (1920, 1), (4, 8)))
    # m_list.append(Modality("co2", [5], (1920, 1), (4, 8)))
    # m_list.append(Modality("rri", [7], (1920, 1), (4, 8)))
    # m_list.append(Modality("amp", [8], (1920, 1), (4, 8)))

    return m_list
