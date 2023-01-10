import keras
import tensorflow as tf
from keras import layers
from keras.layers import Reshape
import tensorflow_addons as tfa

p = 30  # patch numbers
l = 3
ch = 6
num_classes = 1 # For MultiClass num_classes = 2
input_shape = (p * l, ch)
image_size = p * l
patch_size = l
num_patches = p
projection_dim = 32 # 16
num_heads = 4 #4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 3
mlp_head_units = [256, 128]  # [2048, 1024] Size of the dense layers of the final classifier


################################### VIT Transformer ####################################################################

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    normalized_inputs = tfa.layers.InstanceNormalization(axis=-1,
                                                         epsilon=1e-6, center=False, scale=False,
                                                         beta_initializer="glorot_uniform",
                                                         gamma_initializer="glorot_uniform")(inputs)


    reshaped = Reshape((-1, p))(normalized_inputs) #TODO     patches = Patches(patch_size)(normalized_inputs)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(reshaped)

    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1)(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    # representation = tf.keras.layers.GlobalAveragePooling1D()(representation)

    representation = layers.Flatten()(representation)


    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.0)  # change
    logits = layers.Dense(num_classes)(features) # For MultiClass activation='softmax'
    return keras.Model(inputs=inputs, outputs=logits)

########################################################################################################################
if __name__ == "__main__":
    print(create_vit_classifier().summary())