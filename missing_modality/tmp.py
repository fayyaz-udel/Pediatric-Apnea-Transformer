from datetime import datetime
import tensorflow_addons as tfa
from model import *
from util import *
import numpy as np
from TrainMonitor import *
from WarmUpCosine import *


from scipy import signal
data = np.load("D:\\nch_30x128_test.npz", allow_pickle=True)
x, y_apnea, y_hypopnea = data['x'], data['y_apnea'], data['y_hypopnea']

data = x[0]

xx = np.zeros((data.shape[0], 129, 31, 1))
for i in range(data.shape[0]):
    f, t, Zxx = signal.stft(data[i, :, 5], fs=128, padded=False)
    xx[i, :, :, 0] = Zxx


keras.backend.clear_session()

mae_model = MaskedAutoencoder(train_augmentation_model=get_train_augmentation_model(),
                              test_augmentation_model=get_test_augmentation_model(),
                              patch_layer=PatchesPre(), patch_encoder=PatchEncoderPre(),
                              encoder=create_encoder(), decoder=create_decoder(), )


#total_steps = int((len(x_train) / BATCH_SIZE) * EPOCHS)
#warmup_steps = int(total_steps * 0.15)
#scheduled_lrs = WarmUpCosine(learning_rate_base=LEARNING_RATE, total_steps=total_steps, warmup_learning_rate=0.0,warmup_steps=warmup_steps, )

#timestamp = datetime.utcnow().strftime("%y%m%d-%H%M%S")
#train_callbacks = [keras.callbacks.TensorBoard(log_dir=f"mae_logs_{timestamp}")]  #,TrainMonitor(test_ds, epoch_interval=5),]

#optimizer = tfa.optimizers.AdamW(learning_rate=scheduled_lrs, weight_decay=WEIGHT_DECAY)



mae_model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=["mae"])
# mae_model.build(input_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), output_shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
# mae_model.summary()
history = mae_model.fit(x = xx, y = xx, epochs=EPOCHS) #  , validation_data=val_ds, callbacks=train_callbacks, )