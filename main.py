from test import test
from train import train
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# "EOG LOC-M2 0
# "EOG ROC-M1 1

# "EEG F3-M2 2
# "EEG F4-M1 3
# "EEG C3-M2 4
# "EEG C4-M1 5
# "EEG O1-M2 6
# "EEG O2-M1 7
# "EEG CZ-O1 8
# "ECG EKG2-EKG 9
# "RESP PTAF 10
# "RESP AIRFLOW 11
# "RESP THORACIC 12
# "RESP ABDOMINAL 13
# SPO2 14

# RATE 15
# CAPNO 16

# RESP RATE 17

# RRI 18
# R amplitude 19

# Demographic data 20

# sig_dict = {
#     "EOG": [0, 1],
#     "RESP": [2, 3, 4, 5, 8],
#     "SPO2CO2": [6, 7],
#     "ECG": [9, 10],
#     "DEMO": [11]
# }
# "EOG LOC-M2",  # 0
# "EOG ROC-M1",  # 1

# "EEG F3-M2",  # 2
# "EEG F4-M1",  # 3
# "EEG C3-M2",  # 4
# "EEG C4-M1",  # 5
# "EEG O1-M2",  # 6
# "EEG O2-M1",  # 7
# "EEG CZ-O1",  # 8

# "SPO2",  # 9
# "CAPNO",  # 10


# RRI 11
# AMP 12
# DEMO 13

sig_dict = {"EOG": [0, 1],
            "EEG": [2, 3],  # ,6
            "RESP": [7, 8],
            "SPO2CO2": [9, 10],
            "ECG": [11, 12],
            "DEMO": [13],
            "ALL": [7, 9, 10]}
# sig_dict = {
#     "EOG": [0, 1],
#     "EEG": [2,3,4,5],#,6,7,8],
#     "SPO2CO2": [9, 10],
#     "ECG": [11, 12],
#     "DEMO": [13]
# }

channel_list = [
    # ["RESP", "EEG"]

    # ["RESP", "EEG", "SPO2CO2", "ECG"],
    # ["RESP", "EEG", "SPO2CO2", "EOG"],
    ["ALL"],

]

for ch in channel_list:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict[name]
    config = {
        "data_path": "C:\\Data\\finalEEG.npz",
        "model_path": "./weights/Trans_reg_imbal/f",
        "model_name": "100" ,
        "regression": False,

        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 20,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 200,  # best 200
        "channels": chs,
    }
    train(config)
    test(config)
    tf.keras.backend.clear_session()
