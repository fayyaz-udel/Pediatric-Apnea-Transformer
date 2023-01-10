import keras

from test import test
from train import train

DATA_PATH = "C:\\Data\\filtered_balanced.npz"
MODEL_PATH = "./weights/Trans_reg_imbal/f"

config = {
    "model_name": "Transformer_12",
    "regression": False,

    "transformer_layers": 10,  # best
    "drop_out_rate": 0.05,
    "num_patches": 20,
    "transformer_units": 32,  # best
    "regularization_weight": 0.001,  # best
    "num_heads": 4,
    "epochs": 100,  # best
}
train(DATA_PATH, MODEL_PATH, config)
test(DATA_PATH, MODEL_PATH, config)

config = {
    "model_name": "Transformer_13",
    "regression": False,

    "transformer_layers": 10,  # best
    "drop_out_rate": 0.05,
    "num_patches": 30,
    "transformer_units": 32,  # best
    "regularization_weight": 0.001,  # best
    "num_heads": 4,
    "epochs": 100,  # best
}
train(DATA_PATH, MODEL_PATH, config)
test(DATA_PATH, MODEL_PATH, config)

