import os

from test import test
from train import train, train_by_fold

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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

# sig_dict = {"EOG": [0, 1],
#             "EEG": [2, 3],  # ,6
#             "RESP": [7, 8],
#             "SPO2CO2": [9, 10],
#             "ECG": [11, 12],
#             "DEMO": [13],
#             "ALL": [7, 9, 10]}
# sig_dict = {
#     "EOG": [0, 1],
#     "EEG": [2,3,4,5],#,6,7,8],
#     "SPO2CO2": [9, 10],
#     "ECG": [11, 12],
#     "DEMO": [13]
# }

# channel_list = [
#     # ["RESP", "EEG"]
#
#     # ["RESP", "EEG", "SPO2CO2", "ECG"],
#     # ["RESP", "EEG", "SPO2CO2", "EOG"],
#     # ["ALL"],
# ]
########################################### CHAT ###########################################
sig_dict_chat = {
    "EOG": [0, 1],
    "EOG1": [0],
    "EOG2": [1],
    "EEG": [2, 3, 4, 5, 6, 7, 8, 9],
    "EEGF3M2": [2, 7],
    "EEGF4M1": [3, 6],
    "EEGC3M2": [4, 7],
    "EEGC4M1": [5, 6],
    "EEGO1M2": [8, 7],
    "EEGO2M1": [9, 6],
    "ECG": [ 17, 18],
    "Resp": [12, 13, 14],
    "Respairflow": [12],
    "Respadominal": [13],
    "Respthoracic": [14],
    "SPO2CO2": [15, 16],
    "SPO2": [15],
    "CAPNO": [16],
}

channel_list_chat = [
    ["ECG", "SPO2"],



]

# 'E1', # 0
# 'E2', # 1
# 'F3', # 2
# 'F4', # 3
# 'C3', # 4
# 'C4', # 5
# 'M1', # 6
# 'M2', # 7
# 'O1', # 8
# 'O2', # 9
# 'ECG3', # 10
# 'ECG1', # 11

# 'AIRFLOW', # 12
# 'ABD', # 13
# 'CHEST', # 14
# 'SAO2', # 15
# 'CAP',# 16
# RRI # 17
# R amplitude # 18


for ch in channel_list_chat:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict_chat[name]
    print(chstr, chs)
    config = {
        "data_path": "C:\\Data\\chat.npz",
        "model_path": "./weights/chat100/f",
        "model_name": "Transformer_new"+ chstr,
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
