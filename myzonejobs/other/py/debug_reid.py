import os.path as osp
import numpy as np
import skimage
import skimage.io as io
import skimage.transform as transform
import skimage.color as color
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

from myzonecv.apis import inspect as I


def show_img(img):
    plt.axis('off')
    plt.imshow(img)
    plt.show()


data_dir = './workspace/data_zoo/trainval/MyZoneUFC-ReID/MyZoneUFC-ReID_justin_clean_kpts-seg1/train'
ann_file = osp.join(data_dir, 'annotations.json')
img_dir = osp.join(data_dir, 'images')

data_debuger = I.DataInspector('reid/reid_resnet50_256x192_c16_gpool-avgp-pfirst-d0_recon', ann_file, img_dir, 'train_with_recon')
data_debuger.dataset.initialize()
item_pipe = data_debuger.dataset.pipe_item()

d0, lab0 = next(item_pipe)
d1, lab1 = next(item_pipe)
