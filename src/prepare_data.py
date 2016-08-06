from __future__ import print_function, division

import os
import numpy as np
import datetime

import cv2
import h5py

data_path = '../data'

image_rows = 420
image_cols = 580


def create_train_data():
    train_data_path = os.path.join(data_path, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2  # Assumption is that half images are masks

    #  Predefining arrays for train and train_mask
    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0

    print('[{}] Creating training images...'.format(str(datetime.datetime.now())))

    train_ids = []

    for image_name in images:
        if 'mask' in image_name:
            continue

        image_mask_name = image_name.split('.')[0] + '_mask.tif'

        train_id = int(image_name.split('_')[0])

        train_ids += [train_id]

        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('[{}] Loading done.'.format(str(datetime.datetime.now())))

    f = h5py.File(os.path.join(data_path, 'train.h5'), 'w')

    f['train'] = imgs
    f['train_mask'] = imgs_mask
    f['train_ids'] = train_ids

    print('[{}] Saving to .h5 files done.'.format(str(datetime.datetime.now())))
    f.close()


def load_train_data():
    f = h5py.File(os.path.join(data_path, 'train.h5'), 'r')

    imgs_train = np.array(f['train'])
    imgs_mask_train = np.array(f['train_mask'])
    train_ids = np.array(f['train_ids'])
    f.close()

    return imgs_train[:, 0, :400, 100:500].reshape(5634, 1, 400, 400), \
           imgs_mask_train[:, 0, :400, 100:500].reshape(5634, 1, 400, 400), \
           train_ids


def create_test_data():
    train_data_path = os.path.join(data_path, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0

    print('[{}] Creating test images...'.format(str(datetime.datetime.now())))

    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.'.format(str(datetime.datetime.now())))

    f = h5py.File(os.path.join(data_path, 'test.h5'), 'w')

    f['test'] = imgs
    f['test_id'] = imgs_id

    print('[{}] Saving to .h5 files done.'.format(str(datetime.datetime.now())))


def load_test_data():
    f = h5py.File(os.path.join(data_path, 'test.h5'), 'r')

    imgs_test = np.array(f['test'])
    imgs_id = np.array(f['test_id'])
    f.close()
    return imgs_test[:, 0, :400, 100:500].reshape(5508, 1, 400, 400), imgs_id

if __name__ == '__main__':
    create_train_data()
    create_test_data()
