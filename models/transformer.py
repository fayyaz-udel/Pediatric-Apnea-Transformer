import tensorflow as tf
import tensorflow_addons as tfa



class Patches(tf.keras.layers.Layer):
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



class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, l2_weight):
        super(PatchEncoder, self).__init__()
        self.projection_dim = projection_dim
        self.l2_weight = l2_weight
        self.num_patches = num_patches
        self.projection = tf.keras.layers.Dense(units=projection_dim, kernel_regularizer=tf.keras.regularizers.L2(l2_weight),
                                                bias_regularizer=tf.keras.regularizers.L2(l2_weight))


    def call(self, patch):
        return self.projection(patch)


def mlp(x, hidden_units, dropout_rate, l2_weight):
    for _, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation=None, kernel_regularizer=tf.keras.regularizers.L2(l2_weight), bias_regularizer=tf.keras.regularizers.L2(l2_weight))(x)
        x = tf.nn.gelu(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def create_transformer_model(input_shape, num_patches,
                             projection_dim, transformer_layers,
                             num_heads, transformer_units, mlp_head_units,
                             num_classes, drop_out, reg, l2_weight):
    if reg:
        activation = None
    else:
        activation = 'sigmoid'
    inputs = tf.keras.layers.Input(shape=input_shape)
    patch_size = 180 / num_patches

    normalized_inputs = tfa.layers.InstanceNormalization(axis=-1, epsilon=1e-6, center=False, scale=False,
                                                         beta_initializer="glorot_uniform",
                                                         gamma_initializer="glorot_uniform")(inputs)

    patches = Patches(patch_size=patch_size)(normalized_inputs)
    encoded_patches = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim, l2_weight=l2_weight)(patches)
    for i in range(transformer_layers):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=i * drop_out, kernel_regularizer=tf.keras.regularizers.L2(l2_weight),
            bias_regularizer=tf.keras.regularizers.L2(l2_weight))(x1, x1)
        x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, transformer_units, i * drop_out, l2_weight)
        encoded_patches = tf.keras.layers.Add()([x3, x2])

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    features = mlp(x, mlp_head_units, 0.0, l2_weight)

    logits = tf.keras.layers.Dense(num_classes, kernel_regularizer=tf.keras.regularizers.L2(l2_weight), bias_regularizer=tf.keras.regularizers.L2(l2_weight),
                          activation=activation)(features)

    return tf.keras.Model(inputs=inputs, outputs=logits)
