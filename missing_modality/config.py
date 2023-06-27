import tensorflow as tf

# DATA
BUFFER_SIZE = 1024
BATCH_SIZE = 64 # 256
AUTO = tf.data.AUTOTUNE
NUM_CLASSES = 1

# TRAINING
EPOCHS = 50

# AUGMENTATION
IMAGE_SIZE = 128  # We'll resize input images to this size.
IMAGE_CHANNELS = 1  # Number of channels in the input images
PATCH_SIZE = 64  # Size of the patches to be extract from the input images.
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2
MASK_PROPORTION = 0.25

# ENCODER and DECODER
LAYER_NORM_EPS = 1e-6
ENC_PROJECTION_DIM = 64 # Best: 64
DEC_PROJECTION_DIM = 32 # Best: 32
ENC_NUM_HEADS = 4
ENC_LAYERS = 3
DEC_NUM_HEADS = 4
DEC_LAYERS = 1 # The decoder is lightweight but should be reasonably deep for reconstruction.
ENC_TRANSFORMER_UNITS = [
    ENC_PROJECTION_DIM * 2,
    ENC_PROJECTION_DIM,
]  # Size of the transformer layers.
DEC_TRANSFORMER_UNITS = [
    DEC_PROJECTION_DIM * 2,
    DEC_PROJECTION_DIM,
]