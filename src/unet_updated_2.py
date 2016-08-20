from __future__ import division
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from sklearn.cross_validation import train_test_split

from theano.tensor.nnet.nnet import binary_crossentropy


from prepare_data import load_train_data, load_test_data
import datetime
import keras
# img_rows = 64
# img_cols = 80

# img_rows = 128
# img_cols = 128

img_rows = 80 * 2
img_cols = 80 * 2

smooth = 1.0


def inverse_dice_coef(y_true, y_pred):
    y_true_f = K.batch_flatten(1 - y_true)
    y_pred_f = K.batch_flatten(1 - y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return -K.mean(intersection / union)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# def inverse_dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(1 - y_true)
#     y_pred_f = K.flatten(1 - y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return -(2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#
#
# def _find_threashold(y_true, y_pred, a):
#     """Heavy penalty for predicting even one pixel when there is no mask. Train/test sampled randomly =>
#     percent of zero mask should be similar in train, validation and test =>
#      we need to zero out masks in such a way that percent of zero  masks in prediction would be equal
#      to percent of zero masks in train
#     """
#     percent_non_zero_true = np.mean((np.mean(y_true.astype(int), (2, 3)) > 0))
#     num_pixels = np.prod(y_pred[0].shape)
#     for i in range(num_pixels):
#         percent_non_zero_pred = np.mean(np.sum((y_pred > a).astype(int), (2, 3)) > i)
#         if abs(percent_non_zero_true - percent_non_zero_pred) < 0.001:
#             return i


def dice_coef_np(y_true, y_pred, a):
    result = 0
    for i in range(y_true.shape[0]):
        y_true_f = (y_true[i][0].flatten() > a).astype(int)
        y_pred_f = (y_pred[i][0].flatten() > a).astype(int)

        result += (2.0 * np.dot(y_true_f, y_pred_f) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return result / y_true.shape[0]


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred) + 10 * binary_crossentropy(y_pred, y_true)


def get_unet():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)
    conv1 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    conv3 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    conv4 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)
    conv5 = Convolution2D(512, 3, 3, border_mode='same', init='he_uniform')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)
    conv6 = Convolution2D(256, 3, 3, border_mode='same', init='he_uniform')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)
    conv7 = Convolution2D(128, 3, 3, border_mode='same', init='he_uniform')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)
    conv8 = Convolution2D(64, 3, 3, border_mode='same', init='he_uniform')(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    conv9 = Convolution2D(32, 3, 3, border_mode='same', init='he_uniform')(conv9)
    conv9 = BatchNormalization()(conv9)
    conv9 = keras.layers.advanced_activations.PReLU(init='zero', weights=None)(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef, 'binary_crossentropy'])

    return model


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def _add_transformations(X, y):

    X0 = np.ndarray(X.shape)
    y0 = np.ndarray(y.shape)

    X1 = np.ndarray(X.shape)
    y1 = np.ndarray(y.shape)

    X2 = np.ndarray(X.shape)
    y2 = np.ndarray(y.shape)

    for i in range((X.shape[0])):
        X0[i, 0] = cv2.flip(X[i, 0], 0)
        y0[i, 0] = cv2.flip(y[i, 0], 0)

        X1[i, 0] = cv2.flip(X[i, 0], 1)
        y1[i, 0] = cv2.flip(y[i, 0], 1)

        X2[i, 0] = cv2.flip(cv2.flip(X[i, 0], 0), 1)
        y2[i, 0] = cv2.flip(cv2.flip(y[i, 0], 0), 1)

    return np.vstack([X, X0, X1, X2]), np.vstack([y, y0, y1, y2])
    # return np.vstack([X, X0, X1]), np.vstack([y, y0, y1])


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

    # X_train, X_validation, y_train, y_validation = train_test_split(X_train,
    #                                                                 y_train,
    #                                                                 stratify=((np.mean(y_train, (2, 3)) == 0)
    #                                                                           .astype(int)
    #                                                                           .flatten()),
    #                                                                 test_size=0.25)

    return X_train, X_test, y_train, y_test


def train_and_predict():
    print('[{}] Loading and preprocessing train data...'.format(str(datetime.datetime.now())))

    imgs_train, imgs_mask_train, train_ids = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')

    print('[{}] Creating validation set...'.format(str(datetime.datetime.now())))
    X_train, X_val, y_train, y_val = _train_val_split(imgs_train, imgs_mask_train)

    print('[{}] Augmenting train...'.format(str(datetime.datetime.now())))
    X_train, y_train = _add_transformations(X_train, y_train)
    # print('[{}] Augmenting val...'.format(str(datetime.datetime.now())))
    # X_val, y_val = _add_transformations(X_train, y_train)
    # print('[{}] Augmenting test...'.format(str(datetime.datetime.now())))
    # X_test, y_test = _add_transformations(X_train, y_train)

    # mean = np.mean(X_train)  # mean for data centering

    # X_train -= mean
    # X_train /= std  # We can probably live without this.
    X_train /= 255  # We can probably live without this.
    # X_test -= mean
    # X_test /= std  # We can probably live without this.
    # X_test /= 255  # We can probably live without this.
    # X_val -= mean
    # # X_val /= std  # We can probably live without this.
    X_val /= 255  # We can probably live without this.

    y_train = (y_train.astype(np.float32) / 255).astype(int).astype(float)  # scale masks to [0, 1]
    # y_test = (y_test.astype(np.float32) / 255).astype(int).astype(float)    # scale masks to [0, 1]
    y_val = (y_val.astype(np.float32) / 255).astype(int).astype(float)      # scale masks to [0, 1]

    print
    print '[{}] Num train non zero masks...'.format(np.mean((np.mean(y_train, (2, 3)) > 0).astype(int).flatten()))
    print '[{}] Num val non zero masks...'.format(np.mean((np.mean(y_val, (2, 3)) > 0).astype(int).flatten()))
    # print '[{}] Num test non zero masks...'.format(np.mean((np.mean(y_test, (2, 3)) > 0).astype(int).flatten()))

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet()
    # model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', save_best_only=True)

    print('[{}] Fitting model...'.format(str(datetime.datetime.now())))

    model.fit(X_train, y_train,
              batch_size=8,
              nb_epoch=20,
              verbose=1,
              shuffle=True,
              validation_data=(X_val, y_val),
              # callbacks=[model_checkpoint]
              )

    # y_pred = model.predict(X_test)
    #
    # for a in np.arange(0, 1, 0.1):
    #     score = dice_coef_np(y_test, y_pred, a)
    #     print '[{date}] a = {a}. Score = {score}'.format(date=str(datetime.datetime.now()),
    #                                                                                 score=score,
    #                                                                                 a=a)
    print
    print('[{}] Loading and preprocessing test data...'.format(str(datetime.datetime.now())))

    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')
    # imgs_test -= mean
    imgs_test /= 255
    #
    # print('[{}] Loading saved weights...'.format(str(datetime.datetime.now())))
    #
    # model.load_weights('unet.hdf5')

    print('[{}] Predicting masks on test data...'.format(str(datetime.datetime.now())))

    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
