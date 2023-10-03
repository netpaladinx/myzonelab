import os.path as osp

import torch

from myzonecv.configs import get_config_file
from myzonecv.core.config import Config
from myzonecv.core.registry import DATASETS, MODELS
from myzonecv.core.data import get_dataloader


class DataInspector:
    def __init__(self, config, ann_file, img_dir, data_type='train', batch_size=1, num_workers=0):
        assert isinstance(config, (str, Config))
        if isinstance(config, str):
            config = Config.from_file(get_config_file(config))
        config.merge({
            'data': {
                data_type: {
                    'data_source': {
                        'ann_file': ann_file,
                        'img_dir': img_dir
                    },
                    'data_loader': {
                        'batch_size': batch_size,
                        'num_workers': num_workers
                    }
                }
            }
        })

        data_cfg = config.data
        if data_cfg.has('data_params'):
            data_cfg[data_type].update_at_key('data_params', data_cfg.data_params, overwrite=False)

        dataset_cfg = data_cfg[data_type].to_dict()
        dataloader_cfg = dataset_cfg.pop('data_loader')
        self.dataset = DATASETS.create(dataset_cfg)
        self.dataloader = get_dataloader(self.dataset, dataloader_cfg)


class ModelInspector:
    def __init__(self, config, gpu_id=0, ckpt_path=None):
        assert isinstance(config, (str, Config))
        if isinstance(config, str):
            config = Config.from_file(get_config_file(config))
        if ckpt_path:
            config.merge({
                'model.init_cfg': {
                    'type': 'pretrained',
                    'path': ckpt_path
                }
            })

        device = torch.device('cuda', gpu_id)
        self.model = MODELS.create(config.model).to(device)


if __name__ == '__main__':
    data_dir = './workspace/data_zoo/trainval/person_keypoints_val2017'
    ann_file = osp.join(data_dir, 'annotations.json')
    img_dir = osp.join(data_dir, 'images')

    data_debuger = DataInspector('bboxperson-pg_deeplabv3_r50s8_coco_512x512', ann_file, img_dir)

    item_pipe = data_debuger.train_dataset.pipe_item(0)
    for dct, lab in item_pipe:
        print(lab)
