import numpy as np

from missing_modality.train_test import train_test

if __name__ == "__main__":
    config = {
        "EPOCHS": 100,
        "BATCH_SIZE": 256,
        "MODALS": ["eog", "eeg", "resp", "spo2", "ecg", "co2"],
        "NOISE_CHANCE": 0.25,
        "FOLDS": [0,4],
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



    for data_name in [ ('chat',"/home/hamed/dd/chat_b_30x64_")]: #('chat',"/home/hamedcan/dd/chat_b_30x64__",),
        config["DATA_NAME"] = data_name[0]
        config["DATA_PATH"] = data_name[1]
        for model_name in ['qaf']:
            if model_name == "qaf":
                config["STEP"] = 'multimodal'
            else:
                config["STEP"] = 'unimodal'
            config["MODEL_NAME"] = model_name
            for miss_ratio in [0.1, 0.2, 0.3, 0.4, 0.5]:
                for noise_ratio in [10, 20, 30, 40, 50]:
                    config["log_name"] = config["MODEL_NAME"] + "_miss_" + str(miss_ratio) + "_noisy_" + str(noise_ratio)
                    config["MISS_RATIO"] = miss_ratio
                    config["NOISE_RATIO"] = noise_ratio
                    result = train_test(config)
                    out_str = data_name[0] + ", " + model_name + ", " + str(miss_ratio) + ", " + str(noise_ratio) + ", "
                    out_str += str("%.1f (%.1f)" % (np.mean(result.auroc_list), np.std(result.auroc_list))) + " \n"
                    f = open("./result/miss_noise_chat.txt", "a")
                    f.write(out_str)
                    f.close()