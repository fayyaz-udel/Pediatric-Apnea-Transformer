from missing_modality.main import train

MODEL_NAME = "QAF"

if __name__ == "__main__":
    config = {
        "PHASE": "multimodal",  # unimodal, multimodal
        "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
        # "DATA_PATH" = "/media/hamed/NSSR Dataset/nch_30x64_test_"
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "MISS_RATIO": 0.00,
        "NOISE_CHANCE": 0.0,
        "FOLDS": 1,
        "TRAIN": False,
    }
    for noise_ratio in [0.1, 0.2, 0.3, 0.4, 100]:
        config["MODEL_NAME"] = MODEL_NAME + "_noisy_" + str(noise_ratio)
        config["NOISE_RATIO"] = noise_ratio
        train(config)
