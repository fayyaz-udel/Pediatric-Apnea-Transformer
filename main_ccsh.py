import os

from test import test
from train import train, train_by_fold

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


########################################### CCSH ###########################################
#'LOC',  # 0
#'ROC',  # 1

#'C3',  # 2
#'C4',  # 3
#'A1',  # 4
#'A2',  # 5

#'ECG1',  # 6
#'ECG2',  # 7

#'ABDO EFFORT',  # 8
#'THOR EFFORT',  # 9

#'SPO2',  # 10
# 'AIRFLOW',  # 11

# RRI 12
# AMP 13


sig_dict_ccsh = {
    "EOG": [0, 1],
    "EOG1": [0],
    "EOG2": [1],

    "EEG": [2, 3, 4, 5],

    "ECG": [12,13],

    "Resp": [8,9,11],
    "Respairflow": [11],
    "Respadominal": [8],
    "Respthoracic": [9],


    "SPO2": [10],
}

channel_list_ccshs = [
    # ["EOG"],
    ["ECG"],
    # ["Resp"],
    # ["SPO2"],
    ["EOG", "ECG"],
    ["EOG", "Resp"],
    ["EOG", "SPO2"],
    ["ECG", "Resp"],
    ["ECG", "SPO2"],
    ["Resp", "SPO2"],
    ["EOG", "ECG", "Resp"],
    ["EOG", "ECG", "SPO2"],
    ["EOG", "Resp", "SPO2"],
    ["ECG", "Resp", "SPO2"],
    ["EOG", "ECG", "Resp", "SPO2"],

]

for ch in channel_list_ccshs:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict_ccsh[name]
    print(chstr, chs)
    config = {
        "data_path": "C:\\Data\\ccshs.npz",
        "model_path": "./weights/chat100/f",
        "model_name": "Transformer_ccshs_"+ chstr,
        "regression": False,

        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 200,  # best 200
        "channels": chs,
    }
    train(config)
    test(config)
