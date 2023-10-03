import os.path as osp
import math

import numpy as np
import cv2
import torch

from ...registry import DATASETS
from ...utils import tolist
from .base_dataset import BaseIterableDataset


class StreamMixin:
    def display(self, frames, label_fn=str, out_dir=None):
        if out_dir is not None and self.writer is None:
            out_path = osp.join(out_dir, 'output.mp4')
            self.writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MP4V'), self.out_fps, self.display_size)

        if isinstance(frames, dict):
            frames = sorted(frames.items())
        else:
            frames = [(i, frame) for i, frame in enumerate(frames)]
        n_frames = len(frames)
        n_cols = math.ceil(math.sqrt(n_frames))
        n_rows = math.ceil(n_frames / n_cols)
        factor = max(n_cols, n_rows)
        max_w = self.display_size[0] // factor
        max_h = self.display_size[1] // factor
        out_w = max_w * n_cols
        out_h = max_h * n_rows
        out_frame = np.zeros((out_h, out_w, 3)).astype(np.uint8)
        for i, (key, img) in enumerate(frames):
            label = label_fn(key)
            img_h, img_w = img.shape[:2]
            ratio = min(max_w / img_w, max_h / img_h)
            dst_h, dst_w = int(img_h * ratio), int(img_w * ratio)
            dst_img = cv2.resize(img, (dst_w, dst_h))
            ci = i % n_cols
            ri = i // n_rows
            out_frame[ri * max_h:(ri * max_h + dst_h), ci * max_w:(ci * max_w + dst_w)] = dst_img

        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', out_frame)

        if self.writer is not None:
            self.writer.write(out_frame)


@DATASETS.register_class('stream')
class StreamDataset(BaseIterableDataset, StreamMixin):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        assert self.data_source == 0 or (isinstance(self.data_source, str) and osp.isfile(self.data_source))
        self.display_size = self.data_params.get('display_size', (1920, 1080))  # w, h
        self.out_fps = self.data_params.get('out_fps', 1)
        self.capture = None
        self.healthy = None
        self.writer = None
        self.frame_number = 0

    def initialize(self):
        self.capture = cv2.VideoCapture(self.data_source)
        self.healthy = True
        self.frame_number = 0

    def get_unprocessed_item(self):
        input_dict = {'fno': self.frame_number, 'src': self.data_source}
        quit = False
        if self.healthy:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    self.healthy = False
            else:
                self.healthy = False

        if self.healthy:
            input_dict.update({
                'frame_id': (self.frame_number,),
                'frame': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            })
            self.frame_number += 1

        if cv2.waitKey(1) == ord('q') or not self.healthy:
            quit = True
        return quit, input_dict

    def deinitialize(self):
        self.capture.release()
        self.writer.release()
        print(f"Source {self.data_source} is closed")

    @staticmethod
    def get_source_id(frame_id):
        return 0

    @staticmethod
    def get_frame_no(frame_id):
        return frame_id[0]

    @property
    def num_sources(self):
        if isinstance(self.data_source, (list, tuple)):
            return len(self.data_source)
        return 1

    @property
    def num_workers(self):
        return self.num_sources


@DATASETS.register_class('multi_stream')
class MultiStreamDataset(BaseIterableDataset, StreamMixin):
    def __init__(self, data_source=None, data_params=None, data_transforms=None, data_eval=None, name=None):
        super().__init__(data_source, data_params, data_transforms, data_eval, name)
        self.data_source = tolist(self.data_source)
        assert all(src == 0 or (isinstance(src, str) and osp.isfile(src)) for src in self.data_source)
        self.display_size = self.data_params.get('display_size', (1920, 1080))  # w, h
        self.capture = None
        self.mask = None
        self.healthy = None
        self.frame_number = None

    def initialize(self):
        self.capture = {}
        self.healthy = {}
        self.frame_number = {}
        for i, src in enumerate(self.data_source):
            if self.mask is None or self.mask[i] is True:
                self.capture[i] = cv2.VideoCapture(src)
                self.healthy[i] = True
                self.frame_number[i] = 0

    def get_unprocessed_item(self):
        input_dict = {'fno': [], 'src': [], 'frame_id': [], 'frame': []}
        quit = False
        for i, cap in self.capture.items():
            src = self.data_source[i]
            hea = self.healthy[i]
            fno = self.frame_number[i]
            if hea:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        hea = False
                else:
                    hea = False

            if hea:
                input_dict['fno'].append(fno)
                input_dict['src'].append(src)
                input_dict['frame_id'].append((i, fno))
                input_dict['frame'].append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                fno += 1

            self.healthy[i] = hea
            self.frame_number[i] = fno

        if cv2.waitKey(1) == ord('q') or not any(self.healthy.values()):
            quit = True
        return quit, input_dict

    def deinitialize(self):
        for i, cap in self.capture.items():
            src = self.data_source[i]
            cap.release()
            print(f"Source {src} is closed")

    @property
    def num_sources(self):
        if isinstance(self.data_source, (list, tuple)):
            return len(self.data_source)
        return 1

    @property
    def num_workers(self):
        return self.num_sources

    @property
    def worker_init_fn(self):
        def worker_init(worker_id):
            worker_info = torch.utils.data.get_worker_info()
            dataset = worker_info.dataset
            dataset.mask = [i % worker_info.num_workers == worker_info.id for i in range(dataset.num_sources)]
        return worker_init

    @staticmethod
    def get_source_id(frame_id):
        return frame_id[0]

    @staticmethod
    def get_frame_no(frame_id):
        return frame_id[1]
