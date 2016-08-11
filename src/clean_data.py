from __future__ import division

"""
This script makes train data less noisy in a way:

Finds similar images assigns to these clusters of images max mask
"""

import networkx as nx
import os
import pandas as pd

from PIL import Image
import glob
import pandas as pd
import cv2
import os
import numpy as  np
from pylab import *
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

image_rows = 420
image_cols = 580

data_path = '../data'

train_data_path = os.path.join(data_path, 'train')
images = os.listdir(train_data_path)
total = len(images) / 2

imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

i = 0
print('-'*30)
print('Creating training images...')
print('-'*30)
for image_name in images:
    if 'mask' in image_name:
        continue
    image_mask_name = image_name.split('.')[0] + '_mask.tif'
    img = cv2.imread(os.path.join(train_data_path, image_name), cv2.IMREAD_GRAYSCALE)
    img_mask = cv2.imread(os.path.join(train_data_path, image_mask_name), cv2.IMREAD_GRAYSCALE)

    img = np.array([img])
    img_mask = np.array([img_mask])

    imgs[i] = img
    imgs_mask[i] = img_mask

    if i % 100 == 0:
        print('Done: {0}/{1} images'.format(i, total))
    i += 1
print('Loading done.')


train_ids = [x for x in images if 'mask' not in x]

train = pd.DataFrame()
train['subject'] = map(lambda x: int(x.split('_')[0]), train_ids)
train['filename'] = train_ids
train['image_num'] = map(lambda x: int(x.split('.')[0].split('_')[1]), train_ids)

imgs_flat = np.reshape(imgs, (5635, 420*580))

for subject in train['subject'].unque():
    a = imgs_flat[(train['subject'] == subject).astype(int).values == 1]
    b = squareform(pdist(a))

    graph = []
    for i in range(1, 2000):
        for j in range(i + 1, 120):
            if (b < 5000)[(i, j)]:
                graph += [(i, j)]
    G = nx.Graph()
    G.add_edges_from(graph)
    connected_components = list(map(list, nx.connected_component_subgraphs(G)))
    clusters = pd.DataFrame(zip(range(len(connected_components), connected_components)),
                            columns=['cluster_name', 'components'])
    temp = pd.DataFrame()
    temp['image_num'] = train.loc[(train['subject'] == subject), 'image_num']
    temp['subject'] = subject
