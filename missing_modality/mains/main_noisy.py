from missing_modality.train_test import train, train_test

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "Transformer",
        "DATA_NAME": "nch",
        "STEP": "multimodal",  # unimodal, multimodal
        "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
        # "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_test_",
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "MISS_RATIO": 0.00,
        "NOISE_CHANCE": 0.25,
        "FOLDS": 1,
        "PHASE": "TEST",
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
    for noise_ratio in [10, 20, 30, 40, 50]:
        config["log_name"] = config["MODEL_NAME"] + "_noisy_" + str(noise_ratio)
        config["NOISE_RATIO"] = noise_ratio
        train_test(config)