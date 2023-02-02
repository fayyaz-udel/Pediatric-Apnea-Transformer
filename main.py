from test import test
from train import train

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

# for layer in [6,8]:
#     for p in [20,30,60]:
#         for heads in [4,6,8]:
#             for unit in [32, 64]:
config = {
    "data_path": "C:\\Data\\raw_all_RR.npz",
    "model_path": "./weights/Trans_reg_imbal/f",
    "model_name": "Transformer",
    "regression": False,

    "transformer_layers": 4,  # best 5
    "drop_out_rate": 0.25,  # best 0.25
    "num_patches": 20,  # best 30 TBD
    "transformer_units": 32,  # best 32
    "regularization_weight": 0.001,  # best 0.001
    "num_heads": 4,
    "epochs": 100,  # best 200
    "channels": [11, 14, 18, 19],
}
train(config)
test(config)
