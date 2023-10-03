import sys
import types
import traceback


class Registry:
    def __init__(self, name, create_func=None, parent=None):
        self.name = name

        self.create_func = None
        self.parent = None
        self.init(create_func, parent)

        self.class_dict = {}
        self.children = {}

    def init(self, create_func, parent=None):
        if self.create_func is None:
            if (create_func is None) and (parent is not None) and (parent.create_func is not None):
                create_func = parent.create_func.__func__
            self.create_func = types.MethodType(create_func, self) if create_func else None

        if self.parent is None:
            if parent is not None:
                parent.add_children(self)
            self.parent = parent

    def add_children(self, registry):
        assert registry.name not in self.children
        self.children[registry.name] = registry

    def create(self, cfg, args=None, create_func=None, **kwargs):
        if not cfg:
            return None

        if kwargs:
            ignore = kwargs.pop('_ignore', ())
            if ignore:
                ignore = (ignore,) if not isinstance(ignore, (list, tuple)) else ignore

            if isinstance(cfg, (list, tuple)) and isinstance(cfg[0], dict):
                cfg = [{k: v for k, v in {**c, **kwargs}.items() if k not in ignore} for c in cfg]
            elif isinstance(cfg, dict):
                cfg = {k: v for k, v in {**cfg, **kwargs}.items() if k not in ignore}
            else:
                raise TypeError(f"cfg has invalid type: {type(cfg)}")

        if create_func is not None:
            try:
                return create_func(self, cfg) if args is None else create_func(self, cfg, args)
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                raise type(e)(f'{e} \ncfg: {cfg} \nargs: {args}')
        elif self.create_func is not None:
            try:
                return self.create_func(cfg) if args is None else self.create_func(cfg, args)
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                raise type(e)(f'{e} \ncfg: {cfg} \nargs: {args}')
        else:
            return self.default_create(cfg, args)

    def default_create(self, cfg, args=None):
        kwargs = cfg.copy()
        obj_type = kwargs.pop('type')
        obj_cls = self.get(obj_type)
        if obj_cls is None:
            raise KeyError(f'{obj_type} is not registered in {self.name}')
        try:
            if args is None:
                return obj_cls(**kwargs)
            elif isinstance(args, (list, tuple)):
                return obj_cls(*args, **kwargs)
            else:
                return obj_cls(args, **kwargs)
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            raise type(e)(f'{obj_cls.__name__}: {e}')

    def get(self, key):
        sp = key.split('.', 1)
        if len(sp) == 1:  # get from self
            key = sp[0]
            return self.class_dict.get(key)
        else:
            name, key = sp
            if name in self.children:  # get from children
                return self.children[name].get(key)
            else:  # get from root
                parent = self.parent
                while parent.parent is not None:
                    parent = parent.parent
                return parent.get(name + '.' + key)

    def register_class(self, name=None, cls=None, force=False):
        if isinstance(name, type):
            cls = name
            name = cls.__name__

        if cls is not None:
            self._register_class(name, cls, force)
            return cls

        def _register(cls):
            self._register_class(name, cls, force)
            return cls

        return _register

    def _register_class(self, name, cls, force):
        if name is None:
            name = cls.__name__

        if not force and name in self.class_dict:
            raise KeyError(f'{name} is already registered in {self.name}')

        self.class_dict[name] = cls

    def __len__(self):
        return len(self.class_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        return self.__class__.__name__ + f'(name={self.name}, class_dict={self.class_dict})'


# run, run scheduler, runner
RUNS = Registry('run')
RUN_SCHEDULERS = Registry('run_scheduler')
RUNNERS = Registry('runner')

# dataset, data transform
DATASETS = Registry('dataset')
DATA_TRANSFORMS = Registry('data_transform')
POSE_TRANSFORMS = Registry('pose_transform', parent=DATA_TRANSFORMS)
DETECT_TRANSFORMS = Registry('detect_transform', parent=DATA_TRANSFORMS)
REID_TRANSFORMS = Registry('reid_transform', parent=DATA_TRANSFORMS)

# block, layers
BLOCKS = Registry('block')
CONV_LAYERS = Registry('conv_layer', parent=BLOCKS)
NORM_LAYERS = Registry('norm_layer', parent=BLOCKS)
ACTI_LAYERS = Registry('acti_layer', parent=BLOCKS)
PAD_LAYERS = Registry('pad_layer', parent=BLOCKS)
UPSAMPLE_LAYERS = Registry('upsample_layer', parent=BLOCKS)
WARP_LAYERS = Registry('warp_layer', parent=BLOCKS)

# initializer
INITIALIZERS = Registry('initializer')

# backbone
BACKBONES = Registry('backbone')

# head
HEADS = Registry('head')

# generator
GENERATORS = Registry('generator')

# loss
LOSSES = Registry('loss')
DETECT_LOSSES = Registry('detect_loss', parent=LOSSES)
POSE_LOSSES = Registry('pose_loss', parent=LOSSES)
REID_LOSSES = Registry('reid_loss', parent=LOSSES)

# postprocessor
POSTPROCESSORS = Registry('postprocessor')
DETECT_POSTPROCESSOR = Registry('detect_postprocessor', parent=POSTPROCESSORS)
POSE_POSTPROCESSOR = Registry('pose_postprocessor', parent=POSTPROCESSORS)
REID_POSTPROCESSOR = Registry('reid_postprocessor', parent=POSTPROCESSORS)

# model
MODELS = Registry('model')
DETECT_MODELS = Registry('detect_model', parent=MODELS)
POSE_MODELS = Registry('pose_model', parent=MODELS)
REID_MODELS = Registry('reid_model', parent=MODELS)

# optimizer, lr_scheduler, momentum scheduler, & optimize_scheduler
OPTIMIZERS = Registry('optimizer')
LR_SCHEDULERS = Registry('lr_scheduler')
MOMENTUM_SCHEDULERS = Registry('momentum_scheduler')
OPTIMIZE_SCHEDULERS = Registry('optim_scheduler')
