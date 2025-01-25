import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import *
from root.reverse_search.helpful_functions import image2array


PATH = '../../Resources/training'
TRAIN_DATA = image2array(path=PATH).shape
IMG_COUNT = TRAIN_DATA[0]
print(IMG_COUNT)
IMG_SHAPE = TRAIN_DATA[1:]


def build_deep_autoencoder(img_shape, code_size):
    H, W, C = img_shape
    encoder = tf.keras.models.Sequential()  # инициализация модели
    encoder.add(L.InputLayer(img_shape))  # добавление входного слоя, размер равен размеру изображения
    encoder.add(L.Conv2D(filters=32, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=64, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=128, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Conv2D(filters=256, kernel_size=(3, 3), activation='elu', padding='same'))
    encoder.add(L.MaxPooling2D(pool_size=(2, 2)))
    encoder.add(L.Flatten())
    encoder.add(L.Dense(code_size))

    # decoder
    decoder = tf.keras.models.Sequential()
    decoder.add(L.InputLayer((code_size,)))
    decoder.add(L.Dense(14 * 14 * 256))
    decoder.add(L.Reshape((14, 14, 256)))
    decoder.add(L.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='elu', padding='same'))
    decoder.add(L.Conv2DTranspose(filters=3, kernel_size=(3, 3), strides=2, activation=None, padding='same'))

    return encoder, decoder


class AutoEncoder:

    def __init__(self, img_shape, code_size=32):
        self.encoder, self.decoder = build_deep_autoencoder(img_shape, code_size)
        self.encoder.summary()
        self.decoder.summary()


autoencoder = AutoEncoder(IMG_SHAPE, code_size=32)
