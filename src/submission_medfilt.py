from __future__ import division

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from prepare_data import image_cols, image_rows, load_test_data
import h5py

# def prep(img):
#     img = img.astype('float32')
#     # if np.sum(img.astype(int)) < 340:
#     #     return np.zeros(img.shape)
#     img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
#     # img = cv2.resize(img, (image_cols, image_rows))
#     # img = cv2.resize(img, (400, 400))
#     img = cv2.resize(img, (400, 400))
#     c = np.zeros((image_rows, image_cols))
#
#     # c[:400, 100:500] = img
#     c[:400, 100:500] = img
#
#     return c

def prep(img):
    img = img.astype('float32')
    # if np.sum(img.astype(int)) < 340:
    #     return np.zeros(img.shape)
    # img = cv2.threshold(img, 0.5, 1., cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = (img > 0.5).astype(np.uint8)
    # img = cv2.resize(img, (image_cols, image_rows))
    img = cv2.resize(img, (400, 400))

    # img = cv2.resize(img, (300, 300))
    c = np.zeros((image_rows, image_cols))

    if np.sum(img) > 2500:
        c[:400, 100:500] = img

    return c

def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])

black_list = [10, 168, 1625, 1733, 1907, 2343, 2523, 2775, 3128, 4519, 4582, 5063, 5185, 5459]

def submission():
    # imgs_test, imgs_id_test = load_test_data()
    f = h5py.File('../data/test.h5')
    imgs_id_test = np.array(f['test_id'])
    f.close()

    f = h5py.File('../data/pred_15.h5')
    imgs_test = np.array(f['test_mask'])
    f.close()



    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in tqdm(range(total)):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

    file_name = 'submissions/blend2_15_2500.csv'

    df = pd.DataFrame()
    df['img'] = imgs_id_test
    df['pixels'] = rles

    df.loc[df['img'].isin(black_list), 'pixels'] = ''
    df.to_csv(file_name, index=False)

if __name__ == '__main__':
    submission()
