from __future__ import division
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D


from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cross_validation import train_test_split

from prepare_data import load_train_data, load_test_data
import datetime
import keras
img_rows = 64 * 4
img_cols = 80 * 4

smooth = 1.0

def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-7), loss='mse')

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('[{}] Loading and preprocessing train data...'.format(str(datetime.datetime.now())))
    imgs_train, imgs_mask_train, train_ids = load_train_data()
    print('[{}] Loading and preprocessing test data...'.format(str(datetime.datetime.now())))
    imgs_test, imgs_id_test = load_test_data()

    imgs_train = preprocess(imgs_train)
    imgs_test = preprocess(imgs_test)

    X = np.vstack([imgs_train, imgs_test])

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet()
    # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('[{}] Fitting model...'.format(str(datetime.datetime.now())))

    model.fit(X, X,
              batch_size=8,
              nb_epoch=100,
              verbose=1,
              shuffle=True,
              validation_split=0.2,
              )


if __name__ == '__main__':
    train_and_predict()
