import os.path as osp
from threading import Thread

from ..diagnoser import Diagnoser
from ...utils import plot_images


@Diagnoser.register_diagnosis('pose')
def plot_inputs(ctx, state):
    input_batch = ctx.train_batch if state.mode == 'train' else ctx.val_batch
    input_imgs = input_batch['img_np']
    kpts = input_batch['kpts']
    save_path = osp.join(state.out_dir, f'input_imgs_{state.tag}.jpg')

    Thread(target=plot_images,
           args=(input_imgs, save_path),
           kwargs={'data_format': 'NHWC', 'point_data': kpts},
           daemon=True).start()


@Diagnoser.register_diagnosis('pose')
def plot_heatmaps(ctx, state):
    results = ctx.train_results if state.mode == 'train' else ctx.val_results
    pred_heatmaps = results['pred_heatmaps'][0][:, None]  # 17 x 1 x h x w
    save_path = osp.join(state.out_dir, f'heatmaps_{state.tag}.jpg')

    Thread(target=plot_images,
           args=(pred_heatmaps, save_path),
           kwargs={'data_type': 'NCHW', 'img_format': 'GRAYSCALE', 'normalize': True},
           daemon=True).start()
