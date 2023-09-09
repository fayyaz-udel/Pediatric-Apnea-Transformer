from missing_modality.train_test import train, train_test

if __name__ == "__main__":
    config = {
        "MODEL_NAME": "Transformer",
        "DATA_NAME": "nch",
        "STEP": "multimodal",  # unimodal, multimodal
        "DATA_PATH": "/media/hamed/NSSR Dataset/nch_30x64_",
        # "DATA_PATH": "/home/hamedcan/d/nch_30x64_",
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_RATIO": 0.00,
        "NOISE_CHANCE": 0.0,

        "MISS_RATIO": 0.00,
        "MISS_INDEX": 0,

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

    for miss_indexes in [[0,1]]:
        config["log_name"] = config["MODEL_NAME"] + "_" + config["DATA_NAME"] + "_missModal_" + str(config["MISS_INDEX"])
        config["MISS_INDEX"] = miss_indexes
        train_test(config)

