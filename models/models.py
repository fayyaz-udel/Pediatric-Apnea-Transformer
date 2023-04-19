import keras
from keras import Input, Model
from keras.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, LSTM, Bidirectional, Permute, \
    Reshape, GRU, Conv1D, MaxPooling1D, Activation, Dropout, GlobalAveragePooling1D, multiply, MultiHeadAttention, Add, \
    LayerNormalization, SeparableConvolution1D
from keras.models import Sequential
from keras.activations import relu, sigmoid
from keras.regularizers import l2
import tensorflow_addons as tfa

from .transformer import create_transformer_model, mlp, create_hybrid_transformer_model




def create_10_model(input_shape):
    filter_count = [64, 32, 8]
    model = Sequential()
    for i in range(1):
        model.add(Conv1D(filter_count[i], 128))
        model.add(BatchNormalization())
        model.add(Activation(relu))
        model.add(MaxPooling1D())
        model.add(Dropout(0.1))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.1))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation(relu))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))

    return model


def create_58model(input_a_shape, weight=1e-3):
    input1 = Input(shape=input_a_shape)
    input1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                              beta_initializer="glorot_uniform",
                                              gamma_initializer="glorot_uniform")(input1)
    x1 = Conv1D(16, 128, activation='relu')(input1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)
    x1 = Conv1D(8, 128, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)
    x1 = Conv1D(4, 128, activation='relu')(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling1D()(x1)

    x1 = LSTM(32, return_sequences=True)(x1)
    x1 = LSTM(16, return_sequences=True)(x1)
    x1 = LSTM(4, return_sequences=True)(x1)
    x1 = Flatten()(x1)
    x1 = Dense(32, activation='relu')(x1)
    x1 = Dense(32, activation='relu')(x1)
    outputs = Dense(1, activation='sigmoid')(x1)

    model = Model(inputs=input1, outputs=outputs)
    return model


def create_100model(input_a_shape):
    input1 = Input(shape=input_a_shape)
    input1 = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                              beta_initializer="glorot_uniform",
                                              gamma_initializer="glorot_uniform")(input1)
    x1 = Conv1D(128, 32)(input1)
    x1 = Conv1D(64, 32)(x1)

    # Channel-wise attention module
    concat = x1
    squeeze = GlobalAveragePooling1D()(concat)
    excitation = Dense(64, activation='relu')(squeeze)
    excitation = Dense(32, activation='sigmoid')(excitation)
    excitation = Reshape((1, 32))(excitation)

    for i in range(4):
        x1 = excitation  # LayerNormalization(epsilon=1e-6)(encoded_patches) # TODO
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=32)(x1, x1)
        x2 = Add()([attention_output, excitation])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, [64, 32], 0, 0)  # i *
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    # x = Concatenate()([x, demo])
    features = mlp(x, [128, 64], 0.0, 0)

    logits = Dense(1, activation='sigmoid')(features)
    model = Model(inputs=input1, outputs=logits)
    return model


model_dict = {

    "10": create_10_model((60 * 32, 7)),
    "100": create_100model((60 * 32, 7)),
    "58": create_58model((60 * 32, 7)),
    "hybrid": create_hybrid_transformer_model(),
}


def get_model(config):
    if config["model_name"].split('_')[0] == "Transformer":
        return create_transformer_model(input_shape=(60 * 32, len(config["channels"])),
                                        num_patches=config["num_patches"], projection_dim=config["transformer_units"],
                                        transformer_layers=config["transformer_layers"], num_heads=config["num_heads"],
                                        transformer_units=[config["transformer_units"] * 2,
                                                           config["transformer_units"]],
                                        mlp_head_units=[256, 128], num_classes=1, drop_out=config["drop_out_rate"],
                                        reg=config["regression"], l2_weight=config["regularization_weight"])
    else:
        return model_dict.get(config["model_name"].split('_')[0])


if __name__ == "__main__":
    config = {
        "model_name": "hybrid",
        "regression": False,

        "transformer_layers": 4,  # best 5
        "drop_out_rate": 0.25,
        "num_patches": 20,  # best
        "transformer_units": 32,  # best 32
        "regularization_weight": 0.001,  # best 0.001
        "num_heads": 4,
        "epochs": 100,  # best
        "channels": [14, 18, 19, 20],
    }
    model = get_model(config)
    model.build(input_shape=(1, 60 * 32, 7))
    print(model.summary())
