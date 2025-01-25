import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import *
from root.reverse_search.CAEs.auto_encoder import AutoEncoder
from root.reverse_search.helpful_functions import image2array


PATH = '../../Resources/training'
TRAIN_DATA = image2array(path=PATH)
IMG_SHAPE = TRAIN_DATA.shape[1:]

autoencoder = AutoEncoder(IMG_SHAPE, code_size=32)


inp = L.Input(IMG_SHAPE)
code = autoencoder.encoder(inp)
reconstruction = autoencoder.decoder(code)

tf_autoencoder = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
tf_autoencoder.compile(optimizer="adamax", loss='mse')
tf_autoencoder.fit(x=TRAIN_DATA, y=TRAIN_DATA, epochs=10, verbose=1)

images = TRAIN_DATA
codes = autoencoder.encoder.predict(images)
assert len(codes) == len(images)
