"""
This will be different from other scripts:

Data Augmentation.

step 2 => read trained, tune
"""
from __future__ import division
from keras.models import model_from_json

import cv2
import numpy as np

from keras.optimizers import Adam, Nadam, SGD

from keras import backend as K
from sklearn.cross_validation import train_test_split

from theano.tensor.nnet.nnet import binary_crossentropy
from keras.callbacks import History
from tqdm import tqdm

import image_generator_xy

from prepare_data import load_train_data, load_test_data
import datetime

import pandas as pd

import os
# img_rows = 64
# img_cols = 80

# img_rows = 128
# img_cols = 128
#
img_rows = 80 * 2
img_cols = 80 * 2

smooth = 1.0


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_spec(y_true, y_pred):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2. * K.sum(y_true_f * y_pred_f, axis=1, keepdims=True) + smooth
    union = K.sum(y_true_f, axis=1, keepdims=True) + K.sum(y_pred_f, axis=1, keepdims=True) + smooth
    return K.mean(intersection / union)



def dice_coef_np(y_true, y_pred):
    result = 0
    for i in range(y_true.shape[0]):
        y_true_f = cv2.threshold(y_true[i][0], 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8).flatten()
        y_pred_f = cv2.threshold(y_pred[i][0], 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8).flatten()

        result += (2.0 * np.dot(y_true_f, y_pred_f) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    return result / y_true.shape[0]

# def dice_coef_np(y_true, y_pred):
#     result = 0
#     for i in tqdm(range(y_true.shape[0])):
#         y_true_f = y_true[i][0].flatten().astype(int)
#         y_pred_f = y_pred[i][0].flatten().astype(int)
#
#         result += (2.0 * np.dot(y_true_f, y_pred_f) + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
#     return result / y_true.shape[0]


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred) + binary_crossentropy(y_pred, y_true)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('cache', json_name)).read())
    model.load_weights(os.path.join('cache', weight_name))
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


def train_and_predict():
    print('[{}] Loading and preprocessing train data...'.format(str(datetime.datetime.now())))

    imgs_train, imgs_mask_train, train_ids = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')

    X_train = imgs_train
    y_train = imgs_mask_train


    datagen = image_generator_xy.ImageDataGenerator(
        # featurewise_center=False,
        # featurewise_std_normalization=False,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        # zoom_range=0.1,
        # channel_shift_range=5,
        # shear_range=5,
        # zca_whitening=True,
        horizontal_flip=True,
        vertical_flip=True
    )

    mean = np.mean(X_train)  # mean for data centering

    X_train -= mean
    X_train /= 255  # We can probably live without this.

    y_train = (y_train.astype(np.float32) / 255).astype(int).astype(float)  # scale masks to [0, 1]

    print
    print '[{}] Num train non zero masks...'.format(np.mean((np.mean(y_train, (2, 3)) > 0).astype(int).flatten()))
    # print '[{}] Num test non zero masks...'.format(np.mean((np.mean(y_test, (2, 3)) > 0).astype(int).flatten()))

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = read_model('16_20_2016-08-14-21-36')

    print('[{}] Fitting generator...'.format(str(datetime.datetime.now())))

    history = History()

    datagen.fit(X_train)

    print('[{}] Fitting model...'.format(str(datetime.datetime.now())))
    batch_size = 16
    nb_epoch = 20

    # sgd = SGD(lr=1e-5, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(optimizer=Nadam(lr=1e-6), loss=dice_coef_loss,
                  metrics=[dice_coef, dice_coef_spec, 'binary_crossentropy'])

    model.fit_generator(datagen.flow(X_train,
                                     y_train,
                                     batch_size=batch_size,
                                     shuffle=True),
                                     nb_epoch=nb_epoch,
                                     verbose=1,
                                     # validation_data=(X_val, y_val),
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

    print '[{}] Saving history'.format(str(datetime.datetime.now()))
    save_history(history, suffix)


if __name__ == '__main__':
    train_and_predict()
