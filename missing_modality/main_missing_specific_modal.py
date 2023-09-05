from missing_modality.train import train
from missing_modality.train_baseline import train_baseline

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "cnn",
        "DATA_NAME": "chat",
        "PHASE": "multimodal",  # unimodal, multimodal
        "DATA_PATH": "/media/hamed/CHAT Dataset/chat_b_30x64_",
        # "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "MISS_RATIO": 0.00,
        "NOISE_CHANCE": 0.0,
        "FOLDS": 1,
        "TRAIN": False,
        ########################################################
        "transformer_layers": 5,  # best 5
        "drop_out_rate": 0.25,  # best 0.25
        "num_patches": 30,  # best 30 TBD
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best 200
        "channels": [0, 3, 5, 6, 9, 10, 4],
    }

    for miss_ratio in [0]:#, 0.1, 0.2, 0.3, 0.4, 0.5]:
        config["log_name"] = config["MODEL_NAME"] + "_" + config["DATA_NAME"] + "_miss_" + str(miss_ratio)
        config["MISS_RATIO"] = miss_ratio
        train_baseline(config)
