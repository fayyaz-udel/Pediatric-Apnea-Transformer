import os

from test import test
from train import train

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# 0- E1 - M2
# 1- E2 - M1

# 2- F3 - M2
# 3- F4 - M1
# 4- C3 - M2
# 5- C4 - M1
# 6- O1 - M2
# 7- O2 - M1

# 8- ECG3 - ECG1

# 9- CANULAFLOW
# 10- AIRFLOW
# 11- CHEST
# 12- ABD

# 13- SAO2
# 14- CAP
######### ADDED IN THIS STEP #########
# 15- RRI
# 16 Ramp

sig_dict_chat = {
    "EOG": [0, 1],
    "EEG": [4, 5],
    "ECG": [8,15,16],
    "Resp": [9, 10],
    "SPO2": [13],
    "CO2": [14],
}

channel_list_chat = [
    # ["EOG"],
    # ["EEG"],
    # ["ECG"],
    # ["Resp"],
    # ["SPO2"],
    # ["CO2"],
    # ["EOG", "EEG"],
    # ["EOG", "ECG"],
    # ["EOG", "Resp"],
    # ["EOG", "SPO2"],
    # ["EOG", "CO2"],
    # ["EEG", "ECG"],
    # ["EEG", "Resp"],
    # ["EEG", "SPO2"],
    # ["EEG", "CO2"],
    # ["ECG", "Resp"],
    # ["ECG", "SPO2"],
    # ["ECG", "CO2"],
    # ["Resp", "SPO2"],
    # ["Resp", "CO2"],
    # ["SPO2", "CO2"],
    ["EOG", "EEG", "ECG"],
    ["EOG", "EEG", "Resp"],
    ["EOG", "EEG", "SPO2"],
    ["EOG", "EEG", "CO2"],
    ["EOG", "ECG", "Resp"],
    ["EOG", "ECG", "SPO2"],
    ["EOG", "ECG", "CO2"],
    ["EOG", "Resp", "SPO2"],
    ["EOG", "Resp", "CO2"],
    ["EOG", "SPO2", "CO2"],
    ["EEG", "ECG", "Resp"],
    ["EEG", "ECG", "SPO2"],
    ["EEG", "ECG", "CO2"],
    ["EEG", "Resp", "SPO2"],
    ["EEG", "Resp", "CO2"],
    ["EEG", "SPO2", "CO2"],
    ["ECG", "Resp", "SPO2"],
    ["ECG", "Resp", "CO2"],
    ["ECG", "SPO2", "CO2"],
    ["Resp", "SPO2", "CO2"],
    # ["EOG", "EEG", "ECG", "Resp", "SPO2", "CO2"],

]

for ch in channel_list_chat:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict_chat[name]
    print(chstr, chs)
    config = {
        "data_path": "C:\\Data\\chat_3_64.npz",
        "model_path": "./weights/chat100/f",
        "model_name": "Transformer_chat2_"+ chstr,
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