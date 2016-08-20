"""
This will be different from other scripts:

Data Augmentation.
"""

from __future__ import division
from tqdm import tqdm
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, AtrousConvolution2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam, Nadam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cross_validation import train_test_split

from theano.tensor.nnet.nnet import binary_crossentropy
from keras.callbacks import History

import image_generator_xy
from keras.models import Sequential
from prepare_data import load_train_data, load_test_data
import datetime
import keras
import pandas as pd

import os
# img_rows = 64
# img_cols = 80

# img_rows = 128
# img_cols = 128

img_rows = 80 * 2
img_cols = 80 * 2

# img_rows = 64 * 2
# img_cols = 64 * 2

smooth = 1.0


def dice_coef_spec(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def inverse_dice_coef_spec(y_true, y_pred):
    y_true_f = K.batch_flatten(1 - y_true)
    y_pred_f = K.batch_flatten(1 - y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)

def inverse_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(1 - y_true)
    y_pred_f = K.flatten(1 - y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_np(y_true, y_pred):
    result = 0
    for i in tqdm(range(y_true.shape[0])):
        y_true_f = y_true[i][0].flatten().astype(int)
        y_pred_f = y_pred[i][0].flatten().astype(int)

        result += (2.0 * np.dot(y_true_f, y_pred_f) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return result / y_true.shape[0]


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred) + binary_crossentropy(y_pred, y_true)

def dice_coef_spec_loss(y_true, y_pred):
    return 1 - dice_coef_spec(y_true, y_pred) + binary_crossentropy(y_pred, y_true)

def inverse_dice_coef_loss(y_true, y_pred):
    return 1 - inverse_dice_coef_spec(y_true, y_pred) + binary_crossentropy(y_pred, y_true)


def get_unet4():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)
    conv1 = BatchNormalization(mode=0, axis=1)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    conv2 = BatchNormalization(mode=0, axis=1)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    conv3 = BatchNormalization(mode=0, axis=1)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    conv4 = BatchNormalization(mode=0, axis=1)(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)
    conv5 = BatchNormalization(mode=0, axis=1)(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)


    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)


    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)
    conv6 = BatchNormalization(mode=0, axis=1)(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)
    conv7 = BatchNormalization(mode=0, axis=1)(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)
    conv8 = BatchNormalization(mode=0, axis=1)(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = AtrousConvolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    conv9 = AtrousConvolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)

    # conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    # conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    # conv9 = BatchNormalization(mode=0, axis=1)(conv9)
    # conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    # conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    #

    # model.add(AtrousConvolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    # model.add(keras.layers.advanced_activations.ELU())
    # model.add(ZeroPadding2D((1, 1)))
    # model.add(BatchNormalization(mode=0, axis=1))
    # model.add(AtrousConvolution2D(128, 3, 3, border_mode='valid', init='he_uniform'))
    # model.add(keras.layers.advanced_activations.ELU())
    # model.add(ZeroPadding2D((1, 1)))

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def _train_val_split(imgs_train, imgs_mask_train):
    """
    Partition based on the mask existence in the image

    0.2 => validation
    0.2 => test
    0.6 => train
    :param imgs_mask_train:
    :return:
    """
    mean_mask = (np.mean(imgs_mask_train, (2, 3)) == 0).astype(int).flatten()

    # Subtracting test set
    X_train, X_test, y_train, y_test = train_test_split(imgs_train, imgs_mask_train, stratify=mean_mask, test_size=0.2)

    return X_train, X_test, y_train, y_test


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def extra_data():
    imgs_test, imgs_id_test = load_test_data()
    imgs_test_mask = np.load('masks/imgs_mask_test_2016-08-12-07-00.npy')

    return preprocess(imgs_test), preprocess(imgs_test_mask)


def train_and_predict():
    print('[{}] Loading and preprocessing train data...'.format(str(datetime.datetime.now())))

    imgs_train, imgs_mask_train, train_ids = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')

    print('[{}] Creating validation set...'.format(str(datetime.datetime.now())))
    # X_train = imgs_train
    # y_train = imgs_mask_train
    X_train, X_val, y_train, y_val = _train_val_split(imgs_train, imgs_mask_train)

    # print('[{}] Getting extra data...'.format(str(datetime.datetime.now())))
    # extra_x, extra_y = extra_data()
    #
    # X_train = np.vstack([X_train, extra_x[:4000, :, :, :]])
    # X_val = np.vstack([X_val, extra_x[4000:, :, :, :]])


    datagen = image_generator_xy.ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # zoom_range=0.1,
        # # channel_shift_range=30,
        # shear_range=5,
        # # zca_whitening=True,
        horizontal_flip=True,
        vertical_flip=True
    )

    mean = np.mean(X_train)  # mean for data centering

    X_train -= mean
    X_train /= 255  # We can probably live without this.
    # X_test -= mean
    # X_test /= 255  # We can probably live without this.
    X_val -= mean
    X_val /= 255  # We can probably live without this.

    y_train = (y_train.astype(np.float32) / 255).astype(int).astype(float)  # scale masks to [0, 1]
    # y_test = (y_test.astype(np.float32) / 255).astype(int).astype(float)    # scale masks to [0, 1]
    y_val = (y_val.astype(np.float32) / 255).astype(int).astype(float)      # scale masks to [0, 1]

    # y_train = np.vstack([y_train, extra_y[:4000, :, :, :]]).astype(int).astype(float)
    # y_val = np.vstack([y_val, extra_y[4000:, :, :, :]]).astype(int).astype(float)

    print
    print '[{}] Num train non zero masks...'.format(np.mean((np.mean(y_train, (2, 3)) > 0).astype(int).flatten()))
    print '[{}] Num val non zero masks...'.format(np.mean((np.mean(y_val, (2, 3)) > 0).astype(int).flatten()))
    # print '[{}] Num test non zero masks...'.format(np.mean((np.mean(y_test, (2, 3)) > 0).astype(int).flatten()))

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet4()
    # model = vgg_fcn()
    # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('[{}] Fitting generator...'.format(str(datetime.datetime.now())))

    history = History()

    datagen.fit(X_train)

    print('[{}] Fitting model...'.format(str(datetime.datetime.now())))
    batch_size = 16
    nb_epoch = 20

    model.compile(optimizer=Nadam(lr=1e-3), loss=dice_coef_loss,
                  metrics=[dice_coef, dice_coef_spec, 'binary_crossentropy'])

    model.fit_generator(datagen.flow(X_train,
                                     y_train,
                                     batch_size=batch_size,
                                     shuffle=True),
                                     nb_epoch=nb_epoch,
                                     verbose=1,
                                     validation_data=(X_val, y_val),
                                     samples_per_epoch=len(X_train),
                                     callbacks=[history],
                                     # shuffle=True
                                     )

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    print('[{data}] Saving model. Suffix = {suffix}'.format(data=str(datetime.datetime.now()), suffix=suffix))

    model.compile(optimizer=Nadam(lr=1e-4), loss=dice_coef_loss,
                  metrics=[dice_coef, dice_coef_spec, 'binary_crossentropy'])

    model.fit_generator(datagen.flow(X_train,
                                 y_train,
                                 batch_size=batch_size,
                                 shuffle=True),
                    nb_epoch=nb_epoch,
                    verbose=1,
                    validation_data=(X_val, y_val),
                    samples_per_epoch=len(X_train),
                    callbacks=[history],
                    # shuffle=True
                    )

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    print('[{data}] Saving model. Suffix = {suffix}'.format(data=str(datetime.datetime.now()), suffix=suffix))

    model.compile(optimizer=Nadam(lr=1e-5), loss=dice_coef_loss,
                  metrics=[dice_coef, dice_coef_spec, 'binary_crossentropy'])


    model.fit_generator(datagen.flow(X_train,
                                     y_train,
                                     batch_size=batch_size,
                                     shuffle=True),
                                     nb_epoch=nb_epoch,
                                     verbose=1,
                                     validation_data=(X_val, y_val),
                                     samples_per_epoch=len(X_train),
                                     callbacks=[history],
                                     # shuffle=True
                                     )

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    print('[{data}] Saving model. Suffix = {suffix}'.format(data=str(datetime.datetime.now()), suffix=suffix))

    model.compile(optimizer=Nadam(lr=1e-6), loss=dice_coef_loss,
                  metrics=[dice_coef, dice_coef_spec, 'binary_crossentropy'])

    model.fit_generator(datagen.flow(X_train,
                                     y_train,
                                     batch_size=batch_size,
                                     shuffle=True),
                                     nb_epoch=nb_epoch,
                                     verbose=1,
                                     validation_data=(X_val, y_val),
                                     samples_per_epoch=len(X_train),
                                     callbacks=[history],
                                     # shuffle=True
                                     )

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))

    print
    print('[{}] Loading and preprocessing test data...'.format(str(datetime.datetime.now())))

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= 255

    print('[{}] Predicting masks on test data...'.format(str(datetime.datetime.now())))

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('masks/imgs_mask_test_{suffix}.npy'.format(suffix=suffix), imgs_mask_test)

    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))
    print '[{}] Saving history'.format(str(datetime.datetime.now()))
    save_history(history, suffix)


    val_prediction = model.predict(X_val).astype(int).astype(float)
    print 'binarized_prediction on val = ', dice_coef_np(y_val, val_prediction)
    # test_prediction = model.predict(extra_x).astype(int).astype(float)
    # print 'binarized_prediction on test = ', dice_coef_np(y_val, test_prediction)


if __name__ == '__main__':
    train_and_predict()
