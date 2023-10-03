import os.path as osp
from collections import OrderedDict
import copy

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from ....registry import DATASETS
from ....utils import get_logger, compute_cmc, compute_map, get_if_is, write_numpy, stack_imgs, dump_json, mkdir, save_img
from ....errors import DataModeError
from ...datautils import size_tuple, safe_bbox, bbox_center, bbox_scale, npf
from ...dataconsts import KEYPOINT_FLIP_PAIRS
from ..batch_dataset import IterableBatchDataset
from .myzoneufc_consts import REID_BBOX_PADDING_RATIO
from .myzoneufc_data import MyZoneUFCData
from .myzoneufc_visualize import draw_embeddings

logger = get_logger('myzoneufc_reid')


@DATASETS.register_class('myzoneufc_reid')
class MyZoneUFCReID(IterableBatchDataset):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.input_size = size_tuple(self.data_params['input_size'])  # w, h
        self.input_channels = self.data_params['input_channels']
        self.input_aspect_ratio = self.input_size[0] / self.input_size[1]  # w / h
        self.bbox_padding_ratio = self.data_params.get('bbox_padding_ratio', REID_BBOX_PADDING_RATIO)
        self.keypoint_info = {'flip_pairs_ids': KEYPOINT_FLIP_PAIRS}
        self.use_mask = self.data_params.get('use_mask', False)
        self.use_orig = self.data_params.get('use_orig', False)

        if self.data_mode == 'train':
            self.n_samples_per_id = self.data_params.get('n_samples_per_id', 8)
            self.min_ids_per_batch = self.data_params.get('min_ids_per_batch', 8)
            min_batch_size = self.n_samples_per_id * self.min_ids_per_batch
            assert self.batch_size >= min_batch_size, f"batch_size = {self.batch_size} but min_batch_size = {min_batch_size}"
            self.prob_nonfighter = self.data_params.get('prob_nonfighter', 0.1)
            self.prob_opponent = self.data_params.get('prob_opponent', 0.5)

        self.myzoneufc_data = MyZoneUFCData(**data_source)
        self.input_data = self.load_input_data()

    def load_input_data(self):
        samples = []
        nonfighter_gid = None
        fighter_gids = set()
        gid2samples = {}
        gid2opponent = {}

        for ann in self.myzoneufc_data.list_annotations():
            img_id = ann['image_id']
            img = self.myzoneufc_data.get_img(img_id)
            img_h, img_w = img.get('height'), img.get('width')
            bbox = safe_bbox(ann['bbox'], img_h, img_w, include_x1y1=False)
            center = bbox_center(bbox)
            scale = bbox_scale(bbox, self.input_aspect_ratio, self.bbox_padding_ratio)
            kpts = ann.get('keypoints')

            fighter_id = ann['fighter_id']
            fighter_gid = fighter_id.gid  # global ids used as integer class labels (zero-based)

            sample_dict = {
                'ann_id': ann['id'].tid,
                'img_id': img_id.tid,
                'fighter_gid': fighter_gid,
                'bbox': npf(bbox),
                'center': npf(center),
                'scale': npf(scale),
                'flip_pairs': self.keypoint_info['flip_pairs_ids'],
            }

            if kpts is not None:
                sample_dict['kpts'] = npf(kpts).reshape(-1, 3)

            sample_idx = len(samples)
            samples.append(sample_dict)

            if self.myzoneufc_data.is_nonfighter(fighter_id):
                nonfighter_gid = fighter_gid
            else:
                fighter_gids.add(fighter_gid)

            if fighter_gid not in gid2samples:
                gid2samples[fighter_gid] = []
            gid2samples[fighter_gid].append(sample_idx)

            opponent = self.myzoneufc_data.get_opponent(fighter_id)
            if opponent is not None:
                gid2opponent[fighter_gid] = opponent['id'].gid

        input_data = {
            'samples': samples,
            'nonfighter_gid': nonfighter_gid,
            'fighter_gids': fighter_gids,
            'gid2samples': gid2samples,   # maps fighter gid (including non-fighter) to sample indices
            'gid2opponent': gid2opponent  # maps fighter gid to opponent gid
        }
        logger.info(f'{len(samples)} input items loaded (one item one annotation)')
        return input_data

    @property
    def size(self):
        if self.data_mode == 'train':
            return None
        else:
            return len(self.input_data['samples'])

    def __len__(self):
        if self.data_mode == 'train':
            raise DataModeError("Inapplicable in train mode")
        return len(self.input_data['samples'])

    def initialize(self):
        self.sample_number = 0

    def get_unprocessed_batch(self):
        if self.data_mode == 'train':
            quit, batch = self._get_train_batch()
        else:
            quit, batch = self._get_nontrain_batch()

        input_batch = []
        for input_dict in batch:
            input_dict = copy.deepcopy(input_dict)
            input_dict['img'] = self.myzoneufc_data.read_img(input_dict['img_id'])  # 0 ~ 255 np.uint8 (RGB)
            if self.use_mask:
                input_dict['mask'] = self.myzoneufc_data.get_mask(input_dict['ann_id'])
            if self.use_orig:
                input_dict['orig_img'] = input_dict['img'].copy()

            input_batch.append(input_dict)

        return quit, input_batch

    def _get_train_batch(self):
        samples = self.input_data['samples']
        nonfighter_gid = self.input_data['nonfighter_gid']
        fighter_gids = self.input_data['fighter_gids']
        gid2samples = self.input_data['gid2samples']
        gid2opponent = self.input_data['gid2opponent']
        nonfighter_indices = gid2samples.get(nonfighter_gid, [])
        n_batched_samples = 0
        n_batched_ids = 0
        input_indices = []
        input_batch = []

        if len(nonfighter_indices) > 0 and np.random.rand() < self.prob_nonfighter:
            input_indices.append(np.random.choice(nonfighter_indices, size=self.n_samples_per_id))
            n_batched_samples += self.n_samples_per_id
            n_batched_ids += 1

        rest_gids = fighter_gids.copy()
        while True:
            assert len(rest_gids) > 0, f"Runs out of fighter ids for making a batch"

            f1_gid = np.random.choice(list(rest_gids))
            rest_gids.remove(f1_gid)
            input_indices.append(np.random.choice(gid2samples[f1_gid], size=self.n_samples_per_id))
            n_batched_samples += self.n_samples_per_id
            n_batched_ids += 1
            if n_batched_samples >= self.batch_size:
                break

            f2_gid = gid2opponent.get(f1_gid)
            if f2_gid is not None and f2_gid in rest_gids and np.random.rand() < self.prob_opponent:
                rest_gids.remove(f2_gid)
                input_indices.append(np.random.choice(gid2samples[f2_gid], size=self.n_samples_per_id))
                n_batched_samples += self.n_samples_per_id
                n_batched_ids += 1
                if n_batched_samples >= self.batch_size:
                    break

        assert n_batched_ids >= 2, f"One batch needs to contain 2 ids at least"
        input_indices = np.concatenate(input_indices)
        input_batch = [samples[idx] for idx in input_indices]
        self.sample_number += n_batched_samples
        return False, input_batch

    def _get_nontrain_batch(self):
        samples = self.input_data['samples']
        sample_number = self.sample_number
        batch_size = min(self.batch_size, len(samples) - sample_number)
        input_batch = [samples[sample_number + i] for i in range(batch_size)]
        self.sample_number += batch_size
        quit = batch_size <= 0
        return quit, input_batch

    def deinitialize(self):
        # logger.info(f"{self.sample_number} samples have been generated")
        pass

    def evaluate_begin(self, show_recon_num=0, show_recon_dir=None, ctx=None):
        if not hasattr(self, 'show_recon_set'):
            total_num = len(self)
            total_indices = np.arange(total_num)
            if show_recon_num == -1:
                self.show_recon_num = total_num
                self.show_recon_set = set(total_indices)
            else:
                self.show_recon_num = min(show_recon_num, total_num)
                self.show_recon_set = set(np.random.choice(total_indices, self.show_recon_num, replace=False))
            if show_recon_dir is None:
                show_recon_dir = f'{self.name}_recon'
            self.show_recon_dir = osp.join(ctx.get('work_dir', '.'), osp.basename(show_recon_dir))
            if self.show_recon_num > 0:
                mkdir(self.show_recon_dir, exist_ok=True)

        self.show_recon_idx = 0

    @staticmethod
    def _get_suffix(ctx):
        if ctx.has('epoch'):
            return f'_epoch{ctx.epoch}'
        elif ctx.has('step'):
            return f'_step{ctx.step}'
        else:
            return ''

    def evaluate_step(self, results, batch, ctx=None):
        for i, ann_id in enumerate(results['ann_ids']):
            if self.show_recon_idx not in self.show_recon_set:
                self.show_recon_idx += 1
                continue

            ann = self.myzoneufc_data.get_ann(ann_id)
            img = self.myzoneufc_data.get_img(ann['image_id'])
            img_file_sp = img['file_name'].rsplit('.', 1)
            suffix = self._get_suffix(ctx)
            recon_file = f"{img_file_sp[0]}_{str(ann['id'])}{suffix}.{img_file_sp[1]}"
            recon_path = osp.join(self.show_recon_dir, recon_file)
            recon_pred = results['recon_pred'][i]
            recon_gt = results['recon_gt'][i]
            recon_pair = np.concatenate((recon_gt, recon_pred), axis=1)
            save_img(recon_pair, recon_path)
            self.show_recon_idx += 1

    def _process_all_results(self, all_results, output_dir=None, tensorboard_dir=None, show_features=False):
        ann_ids = [ann_id for res_dict in all_results for ann_id in res_dict['ann_ids']]
        fighter_gids = np.concatenate([res_dict['fighter_gids'] for res_dict in all_results])
        output_mat = np.concatenate([res_dict['output_results'] for res_dict in all_results], 0)
        features_mats = OrderedDict()
        for name in all_results[0].get('features_results', []):
            features_mats[name] = np.concatenate([res_dict['features_results'][name] for res_dict in all_results], 0)

        sorted_indices = np.argsort(fighter_gids)
        ann_ids = [ann_ids[idx] for idx in sorted_indices]
        fighter_gids = fighter_gids[sorted_indices]
        counts = np.unique(fighter_gids, return_counts=True)[1]

        output_mat = output_mat[sorted_indices]
        for name, feature in features_mats.items():
            features_mats[name] = feature[sorted_indices]

        if output_dir:
            write_numpy((fighter_gids, output_mat), osp.join(output_dir, 'output.txt'))
            draw_embeddings(output_mat, osp.join(output_dir, 'output_2d.jpg'), labels=fighter_gids, proj_dims=2, figsize=(24, 24))
            draw_embeddings(output_mat, osp.join(output_dir, 'output_3d.jpg'), labels=fighter_gids, proj_dims=3, figsize=(24, 24))

            if show_features:
                for name, feature in features_mats.items():
                    write_numpy((fighter_gids, feature), osp.join(output_dir, f'feature_{name}.txt'))
                    draw_embeddings(feature, osp.join(output_dir, f'feature_{name}_2d.jpg'), labels=fighter_gids, proj_dims=2, figsize=(24, 24))
                    draw_embeddings(feature, osp.join(output_dir, f'feature_{name}_3d.jpg'), labels=fighter_gids, proj_dims=3, figsize=(24, 24))

        if tensorboard_dir:
            writer = SummaryWriter(tensorboard_dir)
            image_ids = [ann['image_id'] for ann in self.myzoneufc_data.get_anns(ann_ids)]
            images = stack_imgs([self.myzoneufc_data.read_img(image_id) for image_id in image_ids], squared=True, size=(100, 100))

            fighter_labels = [str(gid) for gid in fighter_gids]
            writer.add_embedding(output_mat, metadata=fighter_labels, label_img=images, tag='output')
            if show_features:
                for name, feature in features_mats.items():
                    writer.add_embedding(feature, metadata=fighter_labels, label_img=images, tag=f'feature_{name}')
            writer.close()

        return output_mat, features_mats, counts

    def _eval_mat_results(self, output_mat, features_mats, counts, eval_features=True):
        output_dist = np.linalg.norm(output_mat[:, None] - output_mat, ord=2, axis=-1)
        features_dists = OrderedDict()
        if eval_features:
            for name, feature in features_mats.items():
                features_dists[name] = np.linalg.norm(feature[:, None] - feature, ord=2, axis=-1)

        eval_results = OrderedDict()

        # eval cmc
        rank_k = (1, 3, 5)
        output_cmc = compute_cmc(output_dist, counts, counts, rank_k=rank_k)
        for i, k in enumerate(rank_k):
            head = f"Output CMC (rank-{k}): "
            logger.info(f"{head:<50}{output_cmc[i]}")
            eval_results[f'output_cmc_rank{k}'] = output_cmc[i]

        for name, feature_dist in features_dists.items():
            feature_cmc = compute_cmc(feature_dist, counts, counts, rank_k=rank_k)
            for i, k in enumerate(rank_k):
                head = f"Feature '{name}' CMC (rank-{k}): "
                logger.info(f"{head:<50}{feature_cmc[i]}")
                eval_results[f'feature_{name}_cmc_rank{k}'] = feature_cmc[i]

        # eval mAP
        output_map = compute_map(output_dist, counts, counts, query_is_gallery=True)
        head = "Output mAP: "
        logger.info(f"{head:<50}{output_map}")
        eval_results['output_map'] = output_map

        for name, feature_dist in features_dists.items():
            feature_map = compute_map(feature_dist, counts, counts, query_is_gallery=True)
            head = f"Feature '{name}' mAP: "
            logger.info(f"{head:<50}{feature_map}")
            eval_results[f'feature_{name}_map'] = feature_map

        return eval_results

    def evaluate_all(self, all_results, summary_path=None, output_dir=None, tensorboard_dir=None, ctx=None):
        summary_path = get_if_is(ctx, 'summary_path', summary_path, None)
        output_dir = get_if_is(ctx, 'output_dir', output_dir, None)
        tensorboard_dir = get_if_is(ctx, 'tensorboard_dir', tensorboard_dir, None)
        show_features = ctx.get('show_features', False)
        eval_features = ctx.get('eval_features', True)

        output_mat, features_mats, counts = self._process_all_results(all_results, output_dir=output_dir, tensorboard_dir=tensorboard_dir, show_features=show_features)
        eval_results = self._eval_mat_results(output_mat, features_mats, counts, eval_features=eval_features)
        if summary_path:
            dump_json(eval_results, summary_path)
        return eval_results
