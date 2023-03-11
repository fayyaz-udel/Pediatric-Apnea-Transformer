import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Dense, Dropout, Reshape, LayerNormalization, MultiHeadAttention, Add, Flatten, Input, Layer, \
    GlobalAveragePooling1D, AveragePooling1D, Concatenate
from keras.regularizers import L2


gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
tf.keras.backend.clear_session()

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, input):
        input = input[:, tf.newaxis, :, :]
        batch_size = tf.shape(input)[0]
        patches = tf.image.extract_patches(
            images=input,
            sizes=[1, 1, self.patch_size, 1],
            strides=[1, 1, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches,
                             [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim, l2_weight):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.l2_weight = l2_weight
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim, kernel_regularizer=L2(l2_weight),
                                bias_regularizer=L2(l2_weight))
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) # + self.position_embedding(positions)
        return encoded


def mlp(x, hidden_units, dropout_rate, l2_weight):
    for _, units in enumerate(hidden_units):
        x = Dense(units, activation=None, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight))(x)
        x = tf.nn.gelu(x)
        x = Dropout(dropout_rate)(x)
    return x


def create_transformer_model(input_shape, num_patches,
                             projection_dim, transformer_layers,
                             num_heads, transformer_units, mlp_head_units,
                             num_classes, drop_out, reg, l2_weight, demographic=False):
    if reg:
        activation = None
    else:
        activation = 'sigmoid'
    inputs = Input(shape=input_shape)
    patch_size = input_shape[0] / num_patches
    if demographic:
        normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                             beta_initializer="glorot_uniform",
                                                             gamma_initializer="glorot_uniform")(inputs[:,:,:-1])
        demo = inputs[:, :12, -1]

    else:
        normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                             beta_initializer="glorot_uniform",
                                                             gamma_initializer="glorot_uniform")(inputs)

    # patches = Reshape((num_patches, -1))(normalized_inputs)
    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)
    for i in range(transformer_layers):
        x1 = encoded_patches # LayerNormalization(epsilon=1e-6)(encoded_patches) # TODO
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=drop_out, kernel_regularizer=L2(l2_weight),  # i *
            bias_regularizer=L2(l2_weight))(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, drop_out, l2_weight)  # i *
        encoded_patches = Add()([x3, x2])

    x = LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = GlobalAveragePooling1D()(x)
    #x = Concatenate()([x, demo])
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = Dense(num_classes, kernel_regularizer=L2(l2_weight), bias_regularizer=L2(l2_weight),
                   activation=activation)(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)
