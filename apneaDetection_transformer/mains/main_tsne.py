import os

from misc.tsne_test import test_tsne

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

    ["Resp", "SPO2", "CO2"],

]

for ch in channel_list_chat:
    chs = []
    chstr = ""
    for name in ch:
        chstr += name
        chs = chs + sig_dict_chat[name]
    print(chstr, chs)
    config = {
        "data_path": "D:\\Data\\chat_3_64.npz",
        "model_path": "./weights/chat1000/f",
        "model_name": "Transformer_chat22_"+ chstr,
        "regression": False,

        "transformer_layers": 3,  # best 5
        "drop_out_rate": 0.05,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 16,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 2,
        "epochs": 100,  # best 200
        "channels": chs,
    }
    # train(config)
    test_tsne(config)
