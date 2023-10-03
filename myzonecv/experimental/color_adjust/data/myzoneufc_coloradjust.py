import os.path as osp
import random
import copy
from collections import OrderedDict

import numpy as np

from myzonecv.core.registry import DATASETS
from myzonecv.core.utils import get_logger, mkdir, to_img_np, save_img, get_if_is, to_numpy
from myzonecv.core.data.datasets import BaseDataset, MyZoneUFCData
from myzonecv.core.data.datautils import size_tuple
from myzonecv.core.data.dataconsts import REID_BBOX_PADDING_RATIO
from .coloradjust_consts import (IMAGE_STATS, RED_MEAN, RED_STD, GREEN_MEAN, GREEN_STD, BLUE_MEAN, BLUE_STD,
                                 BRIGHTNESS, CONTRAST, SATURATION_MEAN, SATURATION_STD)
from .coloradjust_visualize import draw_histograms

logger = get_logger('myzoneufc_coloradjust')

STATS_META = (('red_mean', RED_MEAN),
              ('red_std', RED_STD),
              ('green_mean', GREEN_MEAN),
              ('green_std', GREEN_STD),
              ('blue_mean', BLUE_MEAN),
              ('blue_std', BLUE_STD),
              ('brightness', BRIGHTNESS),
              ('contrast', CONTRAST),
              ('saturation_mean', SATURATION_MEAN),
              ('saturation_std', SATURATION_STD))
OPS_META = ('contrast', 'bias', 'red', 'green', 'blue', 'saturation')


@DATASETS.register_class('myzoneufc_coloradjust')
class MyZoneUFCColorAdjust(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_size = size_tuple(self.data_params['input_size'])  # w, h
        self.input_channels = self.data_params['input_channels']
        self.input_stats = self.data_params.get('input_stats', IMAGE_STATS)
        self.input_aspect_ratio = self.input_size[0] / self.input_size[1]  # w/h
        self.bbox_padding_ratio = self.data_params.get('bbox_padding_ratio', REID_BBOX_PADDING_RATIO)  # 1. (no padding)

        self.myzoneufc_data = MyZoneUFCData(**self.data_source)
        self.input_data = self.load_input_data()
        self.input_indices = list(range(len(self.input_data)))
        if self.data_mode == 'train' or self.shuffle:
            random.shuffle(self.input_indices)

    def load_input_data(self):
        input_data = []
        for ann in self.myzoneufc_data.list_annotations():
            img_id = ann['image_id']

            input_dict = {
                'ann_id': ann['id'].tid,
                'img_id': img_id.tid
            }
            input_data.append(input_dict)

        logger.info(f'{len(input_data)} input items loaded (one item one image')
        return input_data

    def get_unprocessed_item(self, idx):
        idx = self.input_indices[idx]
        input_dict = copy.deepcopy(self.input_data[idx])
        input_dict['img'] = self.myzoneufc_data.read_img(input_dict['img_id'])  # 0~255 np.uint (RGB)
        return input_dict

    def evaluate_begin(self, show_adjust_num=-1, show_adjust_dir=None, ctx=None):
        if not hasattr(self, 'show_adjust_set'):
            total_num = len(self.input_data)
            if show_adjust_num == -1:
                self.show_adjust_num = total_num
                self.show_adjust_set = set(np.arange(total_num))
            else:
                self.show_adjust_num = min(show_adjust_num, total_num)
                self.show_adjust_set = set(np.random.choice(np.arange(total_num), self.show_adjust_num, replace=False))
            if show_adjust_dir is None:
                show_adjust_dir = f'{self.name}_adjust'
            self.show_adjust_dir = osp.join(ctx.get('work_dir', '.'), osp.basename(show_adjust_dir))
            if self.show_adjust_num > 0:
                mkdir(self.show_adjust_dir, exist_ok=True)

        self.show_adjust_idx = 0

    @staticmethod
    def _get_suffix(ctx):
        if ctx.has('epoch'):
            return f'_epoch{ctx.epoch}'
        elif ctx.has('step'):
            return f'_step{ctx.step}'
        else:
            return ''

    def evaluate_step(self, results, batch, ctx=None):
        op_params_np = results['op_params_np']
        contrast = op_params_np['contrast']
        bias = op_params_np['bias']
        red = op_params_np['red']
        green = op_params_np['green']
        blue = op_params_np['blue']
        saturation = op_params_np['saturation']

        for i, (in_img, out_img, out_img_np, img_id) in enumerate(zip(results['in_img'],
                                                                      results['out_img'],
                                                                      results['out_img_np'],
                                                                      batch['img_id'])):
            if self.show_adjust_idx not in self.show_adjust_set:
                self.show_adjust_idx += 1
                continue

            ops = f'_c{contrast[i]:.2f}b{bias[i]:.2f}-r{red[i]:.2f}g{green[i]:.2f}b{blue[i]:.2f}-s{saturation[i]:.2f}'
            img = self.myzoneufc_data.get_img(img_id)
            img_file_sp = img['file_name'].rsplit('.', 1)
            img_file = f"{img_file_sp[0]}{self._get_suffix(ctx)}{ops}.{img_file_sp[1]}"
            img_path = osp.join(self.show_adjust_dir, img_file)

            in_img = to_img_np(in_img)
            out_img = to_img_np(out_img)
            img_triplet = np.concatenate((in_img, out_img, out_img_np), axis=1)
            save_img(img_triplet, img_path)
            self.show_adjust_idx += 1

    def _process_all_results(self, all_results, output_dir=None, suffix=''):
        in_stats, out_stats, out_stats_np, op_params, op_params_np = [], [], [], [], []
        for res_dict in all_results:
            stats = to_numpy(res_dict['in_stats'])
            in_stats.append(stats)
            stats = np.stack([to_numpy(res_dict['out_stats'][name]) for name in self.input_stats], 1)
            out_stats.append(stats)
            out_stats_np.append(res_dict['out_stats_np'])
            params = np.stack([to_numpy(res_dict['op_params'][name]) for name in OPS_META], 1)
            op_params.append(params)
            params = np.stack([res_dict['op_params_np'][name] for name in OPS_META], 1)
            op_params_np.append(params)
        in_stats = np.concatenate(in_stats)  # N x 10
        out_stats = np.concatenate(out_stats)
        out_stats_np = np.concatenate(out_stats_np)
        op_params = np.concatenate(op_params)
        op_params_np = np.concatenate(op_params_np)

        for i, (target_name, target_value) in enumerate(STATS_META):
            stats_list = (in_stats[:, i], out_stats[:, i], out_stats_np[:, i])
            file_path = osp.join(output_dir, f'summary_on_{target_name}{suffix}.jpg')
            draw_histograms(stats_list,
                            file_path,
                            title=f'summary_on_{target_name}_with_target_{target_value}',
                            subtitles=('in_stats', 'out_stats', 'out_stats_np'),
                            binsize=0.01)
        return in_stats, out_stats, out_stats_np, op_params, op_params_np

    @staticmethod
    def _rmse(a, b):
        return np.sqrt(np.mean(np.square(a - b)))

    def _eval_stats_results(self, in_stats, out_stats, out_stats_np, op_params, op_params_np):
        eval_results = OrderedDict()
        avg_stats_diff = []
        for i, (target_name, target_value) in enumerate(STATS_META):
            eval_results[f'{target_name}_eval_in'] = self._rmse(in_stats[:, i], target_value)
            eval_results[f'{target_name}_eval_out'] = self._rmse(out_stats[:, i], target_value)
            eval_results[f'{target_name}_eval_out_np'] = self._rmse(out_stats_np[:, i], target_value)
            eval_results[f'{target_name}_eval_out_diff'] = self._rmse(out_stats[:, i], out_stats_np[:, i])
            avg_stats_diff.append(eval_results[f'{target_name}_eval_out_np'])
        for i, param_name in enumerate(OPS_META):
            eval_results[f'{param_name}_eval_param_diff'] = self._rmse(op_params[:, i], op_params_np[:, i])
        eval_results['avg_stats_diff'] = sum(avg_stats_diff) / len(avg_stats_diff)
        return eval_results

    def evaluate_all(self, all_results, output_dir=None, ctx=None):
        output_dir = get_if_is(ctx, 'output_dir', output_dir, None)
        if output_dir is None:
            output_dir = f'{self.name}_output'
        output_dir = osp.join(ctx.get('work_dir', '.'), osp.basename(output_dir))
        mkdir(output_dir, exist_ok=True)

        in_stats, out_stats, out_stats_np, op_params, op_params_np = \
            self._process_all_results(all_results, output_dir, self._get_suffix(ctx))
        eval_results = self._eval_stats_results(in_stats, out_stats, out_stats_np, op_params, op_params_np)
        return eval_results
