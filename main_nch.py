import gc

from test import test
from train import train

# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1
#
# "EEG F3-M2",  # 2
# "EEG F4-M1",  # 3
# "EEG C3-M2",  # 4
# "EEG C4-M1",  # 5
# "EEG O1-M2",  # 6
# "EEG O2-M1",  # 7
#
# "ECG EKG2-EKG",  # 8
#
# "RESP PTAF",  # 9
# "RESP AIRFLOW",  # 10
# "RESP THORACIC",  # 11
# "RESP ABDOMINAL",  # 12
# "RESP RATE",  # 13
#
# "SPO2",  # 14
# "CAPNO",  # 15

######### ADDED IN THIS STEP #########
# RRI #16
# Ramp #17
# Demo #18
sig_dict = {"EOG": [0, 1],
            "EEG": [4, 5],
            "RESP": [9, 10],
            "SPO2": [14],
            "CO2": [15],
            "ECG": [8, 16, 17],
            "DEMO": [18],
            }

channel_list = [

    ["EOG", "EEG", "ECG", "RESP", "SPO2","CO2"],
    ["EOG", "EEG", "ECG", "RESP", "SPO2", "CO2", "DEMO"],



]

for ch in channel_list:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict[name]
    config = {
        "data_path": "C:\\Data\\nch2.npz",
        "model_path": "./weights/Trans_reg_imbal/f",
        "model_name": "Transformer_nch2_" + chstr,
        "regression": False,

        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best 200
        "channels": chs,
    }
    train(config)
    test(config)
    gc.collect()
