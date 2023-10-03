import os.path as osp
from threading import Thread

import numpy as np

from ...data.datasets.coco import COCODetect
from ...utils import plot_images, to_numpy
from ..diagnoser import Diagnoser


@Diagnoser.register_diagnosis('detect')
def update_model_config(ctx, diagnoser):
    if diagnoser.predict_train:
        if ctx.model.train_cfg is None:
            ctx.model.train_cfg = {}
        ctx.model.train_cfg['output_preds'] = True


@Diagnoser.register_diagnosis('detect')
def plot_inputs(ctx, state, diagnoser):
    if not diagnoser.plot_inputs:
        return

    input_batch = ctx.train_batch if state.mode == 'train' else ctx.val_batch
    dataset = ctx.train_dataset if state.mode == 'train' else ctx.val_dataset
    inputs = input_batch['inputs']
    targets = input_batch['targets']
    strides = dataset.strides

    input_imgs = to_numpy(inputs['img'])
    save_path = osp.join(state.out_dir, f'input_imgs_{state.tag}.jpg')

    target_cxywh = targets['target_cxywh']
    target_cij = targets['target_cij']
    target_cnt = targets['target_cnt']
    target_cls = targets['target_cls']
    cxywh_data = []
    cls_data = []
    for i, stride in enumerate(strides):
        cxywh = to_numpy(target_cxywh[i])
        cij = to_numpy(target_cij[i])
        cnt = to_numpy(target_cnt[i])
        cls = to_numpy(target_cls[i])
        cxywh[:, :2] += cij
        cxywh *= stride
        s = 0
        for j, c in enumerate(cnt):
            if i == 0:
                cxywh_data.append(cxywh[s:s + c])
                cls_data.append(cls[s:s + c].astype(int))
            else:
                cxywh_data[j] = np.concatenate((cxywh_data[j], cxywh[s:s + c]), 0)
                cls_data[j] = np.concatenate((cls_data[j], cls[s:s + c].astype(int)), 0)
            s += c

    Thread(target=plot_images,
           args=(input_imgs, save_path),
           kwargs={'data_format': 'NCHW', 'normalize': True, 'box_data': cxywh_data, 'box_text_data': cls_data},
           daemon=True).start()


@Diagnoser.register_diagnosis('detect')
def plot_preds(ctx, state, diagnoser):
    if state.mode == 'train' and diagnoser.predict_train:
        input_batch = ctx.train_batch
        results = ctx.train_results
        inputs = input_batch['inputs']
        img_idx = results['img_idx']
        pred_xyxy = results['xyxy_results']
        pred_cls = results['cls_results']

        input_imgs = to_numpy(inputs['img'])
        xyxy_data = []
        cls_data = []
        for i, idx in enumerate(img_idx):
            while idx > len(xyxy_data):
                xyxy_data.append(np.zeros((0, 4)))

            xyxy = pred_xyxy[i]
            cls = pred_cls[i]

            if isinstance(ctx.train_dataset, COCODetect):
                xyxy = xyxy[cls == 0]  # person only
                cls = cls[cls == 0]
            xyxy_data.append(xyxy)
            cls_data.append(cls.astype(int))

        while input_imgs.shape[0] > len(xyxy_data):
            xyxy_data.append(np.zeros((0, 4)))

        save_path = osp.join(state.out_dir, f'pred_imgs_{state.tag}.jpg')

        Thread(target=plot_images,
               args=(input_imgs, save_path),
               kwargs={'data_format': 'NCHW',
                       'normalize': True,
                       'box_data': xyxy_data,
                       'box_format': 'xyxy',
                       'box_text_data': cls_data},
               daemon=True).start()
    elif state.mode == 'val' and diagnoser.predict_val:
        input_batch = ctx.val_batch
        results = ctx.val_results
        orig_imgs = input_batch['orig_img']
        orig_img_ids = input_batch['img_id']
        img_ids = results['img_ids']
        img_idx = [orig_img_ids.index(img_id) for img_id in img_ids]
        pred_xyxy = results['xyxy_results']
        pred_cls = results['cls_results']

        xyxy_data = []
        cls_data = []
        for i, idx in enumerate(img_idx):
            while idx > len(xyxy_data):
                xyxy_data.append(np.zeros((0, 4)))

            xyxy = pred_xyxy[i]
            cls = pred_cls[i]
            if isinstance(ctx.val_dataset, COCODetect):
                xyxy = xyxy[cls == 0]  # person only
                cls = cls[cls == 0]
            xyxy_data.append(xyxy)
            cls_data.append(cls.astype(int))

        while len(orig_imgs) > len(xyxy_data):
            xyxy_data.append(np.zeros((0, 4)))

        save_path = osp.join(state.out_dir, f'pred_imgs_{state.tag}.jpg')

        Thread(target=plot_images,
               args=(orig_imgs, save_path),
               kwargs={'data_format': 'NHWC',
                       'box_data': xyxy_data,
                       'box_format': 'xyxy',
                       'box_text_data': cls_data},
               daemon=True).start()
