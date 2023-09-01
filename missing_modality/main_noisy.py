from missing_modality.train import train
from missing_modality.train_baseline import train_baseline

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "cnn",
        "PHASE": "multimodal",  # unimodal, multimodal
        # "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
        "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_test_",
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "MISS_RATIO": 0.00,
        "NOISE_CHANCE": 0.25,
        "FOLDS": 1,
        "TRAIN": False,
    }
    for noise_ratio in [0.1, 0.2, 0.3, 0.4, 100]:
        config["log_name"] = config["MODEL_NAME"] + "_noisy_" + str(noise_ratio)
        config["NOISE_RATIO"] = noise_ratio
        train_baseline(config)
