from tensorflow import keras
from keras import layers

from keras import metrics
from config import *
from missing_modality.PatchEncoder import PatchEncoder
from missing_modality.Patches import Patches
from missing_modality.model import create_encoder, MaskedAutoencoder, create_decoder
from missing_modality.util import get_augmentation_model
from scipy import signal
import numpy as np

### DATASET ###
data = np.load("D:\\nch_30x128_test.npz", allow_pickle=True)
x = np.concatenate((data['x'][0],data['x'][1], data['x'][2], data['x'][3], data['x'][4]), axis=0)
y_apnea = np.concatenate((data['y_apnea'][0],data['y_apnea'][1], data['y_apnea'][2], data['y_apnea'][3], data['y_apnea'][4]), axis=0)
y_hypopnea = np.concatenate((data['y_hypopnea'][0],data['y_hypopnea'][1], data['y_hypopnea'][2], data['y_hypopnea'][3], data['y_hypopnea'][4]), axis=0)
y = np.sign(y_apnea + y_hypopnea)


xx = np.zeros((x.shape[0], 128, 30, 1))
for i in range(x.shape[0]):
    f, t, Zxx = signal.stft(x[i, :, 5], fs=128, padded=False)
    Zxx= np.abs(Zxx)[:128, :30]
    xx[i, :, :, 0] = ((Zxx-np.min(Zxx))/(np.max(Zxx)-np.min(Zxx)))

train_ds = tf.data.Dataset.from_tensor_slices((xx))
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)

keras.backend.clear_session()

mae_model = MaskedAutoencoder(
    train_augmentation_model=get_augmentation_model(),
    test_augmentation_model=get_augmentation_model(),
    patch_layer=Patches(),
    patch_encoder=PatchEncoder(),
    encoder=create_encoder(),
    decoder=create_decoder(),
)

mae_model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=["mae"])

history = mae_model.fit(train_ds, epochs=5)

loss, mae = mae_model.evaluate(train_ds)
print(f"Loss: {loss:.2f}")
print(f"MAE: {mae:.2f}")


##################################### EVALUATE #########################################################


########################################################################################################

# Extract the augmentation layers.
train_augmentation_model = mae_model.train_augmentation_model
test_augmentation_model = mae_model.test_augmentation_model

# Extract the patchers.
patch_layer = mae_model.patch_layer
patch_encoder = mae_model.patch_encoder
patch_encoder.downstream = True  # Swtich the downstream flag to True.

# Extract the encoder.
encoder = mae_model.encoder

# Pack as a model.
downstream_model = keras.Sequential(
    [
        layers.Input((IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)),
        patch_layer,
        patch_encoder,
        encoder,
        layers.BatchNormalization(),  # Refer to A.1 (Linear probing)
        layers.GlobalAveragePooling1D(),
        layers.Dense(NUM_CLASSES, activation="sigmoid"),
    ],
    name="linear_probe_model",
)

# Only the final classification layer of the `downstream_model` should be trainable.
# for layer in downstream_model.layers[:-1]:
#     layer.trainable = False

##############################################################################################

train_ds_2 = tf.data.Dataset.from_tensor_slices((xx, y))
train_ds_2 = train_ds_2.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(AUTO)
aug = get_augmentation_model()
train_ds_2 = train_ds_2.map(lambda x, y: (aug(x), y), num_parallel_calls=AUTO)


downstream_model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy", metrics.AUC(), metrics.Precision(), metrics.Recall()])
downstream_model.fit(train_ds_2, epochs=EPOCHS, verbose=1)
downstream_model.summary()
loss, accuracy = downstream_model.evaluate(train_ds_2)
accuracy = round(accuracy * 100, 2)
print(f"Accuracy on the test set: {accuracy}%.")
print(np.sum(y), len(y))
