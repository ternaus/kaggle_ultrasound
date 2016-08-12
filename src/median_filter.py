from __future__ import division
'''
This script performs median filtering for our prediction
'''

from scipy.signal import medfilt
import datetime
import numexpr as ne
import h5py
import os
from joblib import Parallel, delayed
import time
import numpy as np

mask = np.load('../src/masks/imgs_mask_test_2016-08-12-07-00.npy')

data_path = '../data'
kernel = 15
result = []

print '[{}] starting filtering'.format(str(datetime.datetime.now()))
start_time = time.time()

print np.array(mask).shape
X = np.split(np.array(mask), 36)


def helper(x):
    return medfilt(x, (1, 1, kernel, kernel))

y_test_new = np.vstack(Parallel(n_jobs=36)(delayed(helper)(x) for x in X))



print '[{datetime}] finished filtering. Tool {time} seconds'.format(datetime=str(datetime.datetime.now()),
                                                                    time=(time.time()-start_time))



print '[{datetime}] Total mean: {target_value}'.format(datetime=str(datetime.datetime.now()),
                                                    target_value=np.mean(np.sum(y_test_new[:, :, :, :].round(), (2, 3)) == 0))


print '[{datetime}] We want: {target_value}'.format(datetime=str(datetime.datetime.now()),
                                                                 target_value=0.51031)

f = h5py.File(os.path.join(data_path, 'pred_{kernel}.h5'.format(kernel=kernel)), 'w')

f['test_mask'] = y_test_new
# ?f['test_ids'] = train_ids

print('[{}] Saving to .h5 files done.'.format(str(datetime.datetime.now())))
f.close()
