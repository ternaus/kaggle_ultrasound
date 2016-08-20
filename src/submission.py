from __future__ import division

import pandas as pd
import numpy as np
import cv2
from prepare_data import image_cols, image_rows, load_test_data
from tqdm import tqdm

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

    if sum(img) > 3000:
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
    imgs_test, imgs_id_test = load_test_data()
    # 'imgs_mask_test_2016-08-15-01-38.npy'
    # 'imgs_mask_test_2016-08-15-08-44.npy'
    # 'imgs_mask_test_2016-08-15-06-37.npy'
    #
    # imgs_test1 = np.load('split5/imgs_mask_test_2016-08-15-01-38.npy')
    # imgs_test2 = np.load('split5/imgs_mask_test_2016-08-15-08-44.npy')
    # imgs_test3 = np.load('split5/imgs_mask_test_2016-08-15-06-37.npy')
    # #
    # imgs_test_a = (imgs_test1 + imgs_test2 + imgs_test3) / 3
    imgs_test_b = np.load('masks/imgs_mask_test_2016-08-14-22-41.npy')
    imgs_test_c = np.load('masks/imgs_mask_test_2016-08-12-07-00.npy')
    # imgs_test_d = np.load('masks/imgs_mask_test_2016-08-16-00-54.npy')
    # imgs_test_e = np.load('masks/imgs_mask_test_2016-08-15-21-35.npy')
    #
    # imgs_test = (imgs_test_a + imgs_test_b + imgs_test_c + imgs_test_d + imgs_test_e) / 5

    imgs_test = np.load('masks/imgs_mask_test_2016-08-17-00-53.npy')

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

    # file_name = 'submissions/blend5.csv'
    file_name = 'submissions/submission_2016-08-17-00-53.csv'

    df = pd.DataFrame()
    df['img'] = imgs_id_test
    df['pixels'] = rles

    df.loc[df['img'].isin(black_list), 'pixels'] = ''
    df.to_csv(file_name, index=False)

if __name__ == '__main__':
    submission()
