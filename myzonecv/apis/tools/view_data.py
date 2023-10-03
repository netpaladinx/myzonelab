import os
import os.path as osp
from collections import defaultdict
import json

import numpy as np
import cv2
import torch

from myzonecv.configs import get_config_file
from myzonecv.core.config import Config
from myzonecv.core.registry import MODELS
from myzonecv.core.consts import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from myzonecv.core.utils import (mkdir, print_progress, to_tensor, normalize, to,
                                 get_affine_matrix, apply_warp_to_map2d)
from myzonecv.core.data.dataloader import collate
from myzonecv.core.data.datasets.coco import (draw_anns, safe_bbox, bbox_center, bbox_scale, npf,
                                              BBOX_PADDING_RATIO, BBOX_SCALE_UNIT, BORDER_COLOR_VALUE)
from myzonecv.core.data.datasets.coco.coco_eval import compute_oks

MODE_BY_IMAGE = 'by_image'
MODE_BY_ANNOTATION = 'by_annotation'


def view_dataset(dataset_dir,
                 show_bbox=True,
                 show_kpts=True,
                 show_seg=False,
                 display=True,
                 fps=1000,
                 save_dir=None,
                 save_mode=MODE_BY_ANNOTATION,
                 revise_ann=None):
    def _get_img_to_anns(f):
        with open(f) as fin:
            data = json.load(fin)
        imgs = data['images']
        anns = data['annotations']
        id2img = {img['id']: img for img in imgs}
        img2anns = defaultdict(list)
        for ann in anns:
            img = id2img[ann['image_id']]
            img_file = osp.basename(img['file_name'])
            img2anns[img_file].append(ann)
        return img2anns

    def _imwrite(path, image):
        try:
            cv2.imwrite(path, image)
        except Exception:
            print(f"Error occurred when processing image (shape: {image.shape}) to path '{path}'")
            pass

    dataset_name = osp.basename(dataset_dir)
    delay = max(1, 1000 // fps)
    img_dir = osp.join(dataset_dir, 'images')
    ann_file = osp.join(dataset_dir, 'annotations.json')
    img2anns = _get_img_to_anns(ann_file)
    displaying = False
    if save_dir:
        mkdir(save_dir, exist_rm=True)

    img_files = sorted([fname for fname in os.listdir(img_dir) if fname.endswith('.jpg')])
    for i, img_file in enumerate(img_files):
        img_path = osp.join(img_dir, img_file)
        image = cv2.imread(img_path)

        anns = img2anns[img_file]
        if callable(revise_ann):
            anns = [revise_ann(ann, image=image) for ann in anns]

        if display:
            img_display = draw_anns(anns, image.copy(), draw_bbox=show_bbox, draw_kpts=show_kpts, draw_seg=show_seg)

            cv2.imshow(dataset_name, img_display)
            displaying = True

            if cv2.waitKey(delay) == ord('q'):
                break

        if save_dir:
            if save_mode == MODE_BY_ANNOTATION:
                for ann in anns:
                    img_save = draw_anns([ann], image.copy(), draw_bbox=show_bbox, draw_kpts=show_kpts, draw_seg=show_seg)

                    x0, y0, w, h = [int(v) for v in ann['bbox'][:4]]
                    x1, y1 = x0 + w, y0 + h
                    img_h, img_w = img_save.shape[:2]
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
                    ann_img = img_save[y0:y1, x0:x1]
                    ann_id = str(ann['id'])
                    save_path = osp.join(save_dir, img_file.replace('.jpg', f'_{ann_id}.jpg'))
                    _imwrite(save_path, ann_img)
            else:
                img_save = draw_anns(anns, image.copy(), draw_bbox=show_bbox, draw_kpts=show_kpts, draw_seg=show_seg)
                save_path = osp.join(save_dir, img_file)
                _imwrite(save_path, img_save)

        print_progress(i, len(img_files), max_intervals=1000)

    if displaying:
        cv2.destroyAllWindows()


def view_single(img,
                anns,
                dataset_dir=None,
                show_bbox=True,
                show_kpts=True,
                show_seg=False,
                output_mode=MODE_BY_ANNOTATION,
                revise_ann=None):
    if isinstance(img, dict):
        img = img['file_name']
    if isinstance(img, str):
        assert dataset_dir is not None
        img_path = osp.join(dataset_dir, 'images', img)
        img = cv2.imread(img_path)
    assert isinstance(img, np.ndarray)

    if callable(revise_ann):
        anns = [revise_ann(ann, image=img) for ann in anns]

    if output_mode == MODE_BY_ANNOTATION:
        img_out = []
        for ann in anns:
            img_o = draw_anns([ann], img.copy(), draw_bbox=show_bbox, draw_kpts=show_kpts, draw_seg=show_seg)
            img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)

            x0, y0, w, h = [int(v) for v in ann['bbox'][:4]]
            x1, y1 = x0 + w, y0 + h
            img_h, img_w = img_o.shape[:2]
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(img_w, x1), min(img_h, y1)
            img_out.append(img_o[y0:y1, x0:x1])
    else:
        img_out = draw_anns(anns, img.copy(), draw_bbox=show_bbox, draw_kpts=show_kpts, draw_seg=show_seg)
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    return img_out


def test_pose_single(img,
                     anns,
                     dataset_dir=None,
                     model=None,
                     config=None,
                     checkpoint=None,
                     checkpoint_type='checkpoint',
                     input_size=(192, 256),
                     gpu_id=0):
    if isinstance(img, dict):
        img = img['file_name']
    if isinstance(img, str):
        assert dataset_dir is not None
        img_path = osp.join(dataset_dir, 'images', img)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    assert isinstance(img, np.ndarray)

    assert isinstance(gpu_id, int) and gpu_id >= 0
    assert torch.cuda.is_available()
    device = torch.device('cuda', gpu_id)

    if model is None:
        if isinstance(config, str):
            if not osp.exists(config):
                config = get_config_file(config)
            config = Config.from_file(config)
        assert isinstance(config, Config), f'Invalid config: {config}'

        assert osp.isfile(checkpoint), f'Invalid checkpoint: {checkpoint}'
        config.model.init_cfg = {'type': checkpoint_type, 'path': checkpoint}

        model = MODELS.create(config.model).to(device)
        print(f"Built model {config.model.type}")

    inputs = []
    img_h, img_w = img.shape[:2]
    input_aspect_ratio = input_size[0] / input_size[1]
    for ann in anns:
        ann_id = ann['id']
        img_id = ann['image_id']
        bbox = safe_bbox(ann['bbox'], img_h, img_w)
        center = npf(bbox_center(bbox))
        scale = npf(bbox_scale(bbox, input_aspect_ratio, BBOX_PADDING_RATIO))

        mat = get_affine_matrix(center, scale, input_size, scale_unit=BBOX_SCALE_UNIT)
        img_input = apply_warp_to_map2d(img, mat, input_size, flags=cv2.INTER_LINEAR, border_value=BORDER_COLOR_VALUE)

        tensor_input = to_tensor(img_input)
        tensor_input = normalize(tensor_input, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD)

        input_dict = {
            'ann_id': ann_id,
            'img_id': img_id,
            'bbox': bbox,
            'center': center,
            'scale': scale,
            'img_orig': img,
            'img_np': img_input,
            '_inputs': {'img': tensor_input},
            '_meta': {'input_batching': 'stack'}
        }
        inputs.append(input_dict)

    batch_inputs = collate(inputs)
    batch_inputs['inputs'] = to(batch_inputs['inputs'], device, model.dtype)

    with torch.no_grad():
        model.eval()
        results = model.eval_step(batch_inputs['inputs'], batch_inputs)
    kpts_results = results['kpts_results']
    ann_ids = results['ann_ids']

    pred_anns = []
    for i, (ann_id, pred_kpts) in enumerate(zip(ann_ids, kpts_results)):
        gt_ann = anns[i]
        assert ann_id == batch_inputs['ann_id'][i] == gt_ann['id']
        bbox = batch_inputs['bbox'][i]
        area = bbox[2] * bbox[3]
        gt_kpts = npf(gt_ann['keypoints'])
        pred_kpts = pred_kpts.flatten()
        oks, (avg_dist, max_dist) = compute_oks(gt_kpts, pred_kpts, area, area, return_dist=True)

        pred_anns.append({
            'id': gt_ann['id'],
            'category_id': gt_ann['category_id'],
            'image_id': gt_ann['image_id'],
            'bbox': gt_ann['bbox'],
            'area': gt_ann['area'],
            'num_keypoints': gt_ann['num_keypoints'],
            'keypoints': pred_kpts.tolist(),
            'oks': oks,
            'avg_dist': avg_dist,
            'max_dist': max_dist
        })

    return anns, pred_anns, batch_inputs, model
