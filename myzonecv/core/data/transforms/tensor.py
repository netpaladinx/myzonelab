import torch

from ...registry import DATA_TRANSFORMS
from ...consts import IMAGENET_RGB_MEAN, IMAGENET_RGB_STD
from ...utils import tolist, to_tensor, normalize


@DATA_TRANSFORMS.register_class('to_tensor')
class ToTensor:
    def __init__(self, tt_map=None):
        self.tt_map = ('img', *tolist(tt_map, exclude='img'))

    def __call__(self, input_dict, dataset, step):
        for name in self.tt_map:
            if isinstance(input_dict, list):
                input_dict = [self._to_tensor(in_dict, name) for in_dict in input_dict]
            else:
                input_dict = self._to_tensor(input_dict, name)

        return input_dict

    def _to_tensor(self, input_dict, name):
        map_np = input_dict.get(name)
        if map_np is None:
            return input_dict

        # shape:
        #   1) h,w,c => c,h,w
        #   2) h,w => h,w
        #   3) n,h,w,c => n,c,h,w
        # dtype:
        #   1) np.uint8 => torch.float32
        #   2) same dtype
        # value:
        #   1) if np.uint8, value.div(255)
        #   2) same value
        map = to_tensor(map_np)

        input_dict[name + '_np'] = map_np
        input_dict[name] = map
        return input_dict


@DATA_TRANSFORMS.register_class('normalize')
class Normalize:
    def __init__(self, nm_map=None, mean=IMAGENET_RGB_MEAN, std=IMAGENET_RGB_STD):
        self.nm_map = ('img', *tolist(nm_map, exclude='img'))
        self.mean = mean
        self.std = std

    def __call__(self, input_dict, dataset, step):
        for name in self.nm_map:
            if isinstance(input_dict, list):
                input_dict = [self._normalize(in_dict, name) for in_dict in input_dict]
            else:
                input_dict = self._normalize(input_dict, name)

        return input_dict

    def _normalize(self, input_dict, name):
        map = input_dict.get(name)
        if map is None:
            return input_dict

        map = normalize(map, self.mean, self.std)
        input_dict[name] = map
        return input_dict


@DATA_TRANSFORMS.register_class('feed_more')
class FeedMore:
    def __init__(self, feed_map=None):
        self.feed_map = tolist(feed_map, exclude='img')

    def __call__(self, input_dict, dataset, step):
        img = input_dict['img']
        inputs = [img]
        for name in self.feed_map:
            map = input_dict.get(name)
            if map is None:
                continue
            inputs.append(map)

        if len(inputs) > 1:
            input_dict['img'] = torch.cat(inputs, 0)

        return input_dict
