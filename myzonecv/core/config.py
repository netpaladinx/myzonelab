import os.path as osp

from .utils import Dict, load_yaml, dump_yaml, str2int


class ConfigDict(Dict):
    def __missing__(self, name):
        """ Doesn't allow any missing key """
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        except Exception as e:
            ex = e
        raise ex


class Config:
    def __init__(self, cfg_dict=None, path=None):
        object.__setattr__(self, '_cfg_dict', ConfigDict(cfg_dict))
        object.__setattr__(self, '_path', path)

    @property
    def path(self):
        return object.__getattribute__(self, '_path')

    def __setitem__(self, key, val):
        self._cfg_dict[key] = val

    def __setattr__(self, name, val):
        if hasattr(self.__class__, name):
            raise AttributeError(f"Config's object attribute '{name}' is read-only")
        else:
            setattr(self._cfg_dict, name, val)

    def __getitem__(self, key):
        return self._cfg_dict[key]

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __delitem__(self, key):
        del self._cfg_dict[key]

    def __delattr__(self, name):
        return delattr(self._cfg_dict, name)

    def __len__(self):
        return len(self._cfg_dict)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._path)

    def __setstate__(self, state):
        cfg_dict, path = state
        object.__setattr__(self, '_cfg_dict', cfg_dict)
        object.__setattr__(self, '_path', path)

    def __repr__(self):
        return f'Config (path: {self.path}): {repr(self._cfg_dict)}'

    @staticmethod
    def from_file(path):

        def _get_by_key_path(obj, key_path, ignore_rest=False):
            if isinstance(key_path, str):
                key_path = key_path.strip('.').split('.')
            assert isinstance(key_path, (list, tuple))

            for key in key_path:
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                    continue
                if isinstance(obj, (list, tuple)):
                    key = str2int(key)
                    if key is not None:
                        obj = obj[key]
                        continue
                if ignore_rest:
                    return obj
                else:
                    raise KeyError
            return obj

        def _resolve_ref(obj, key_path, root):
            if isinstance(obj, str) and obj.startswith('$'):
                try:
                    val = _get_by_key_path(root, obj[1:])
                    return val
                except KeyError:
                    return obj

            if isinstance(obj, dict):
                for key, val in obj.items():
                    kpath = key_path + '.' + key if key_path else key
                    obj[key] = _resolve_ref(val, kpath, root)
            if isinstance(obj, (list, tuple)):
                for i, elem in enumerate(obj):
                    kpath = key_path + '.' + str(i) if key_path else str(i)
                    obj[i] = _resolve_ref(elem, kpath, root)

            return obj

        path = osp.abspath(osp.expanduser(path))
        if not osp.isfile(path):
            raise FileNotFoundError(f"File not found at path '{path}'")
        ext = osp.splitext(path)[1]
        if ext not in ('.yaml', '.yml'):
            raise IOError('Only yaml/yml file type are supported')

        cfg_dict = load_yaml(path)
        cfg_dict = _resolve_ref(cfg_dict, '', cfg_dict)
        return Config(cfg_dict, path)

    def merge(self, opts, allow_key_chain=True, allow_int_key=True):
        def _merge_dict(d1, d2):
            if not (isinstance(d1, dict) and isinstance(d2, dict)):
                return d2

            for k1, v1 in d1.items():
                if k1 in d2:
                    d1[k1] = _merge_dict(v1, d2[k1])
            for k2, v2 in d2.items():
                if k2 not in d1:
                    d1[k2] = v2
            return d1

        def _process_key_chain(obj, allow_key_chain=True):
            if isinstance(obj, (list, tuple)):
                return type(obj)([_process_key_chain(o, allow_key_chain=allow_key_chain) for o in obj])
            elif not isinstance(obj, dict):
                return obj

            out = ConfigDict()
            for key, val in obj.items():
                keys = key.split('.') if allow_key_chain else [key]
                d = out
                for k in keys[:-1]:
                    d.setdefault(k, ConfigDict())
                    d = d[k]
                k = keys[-1]
                val = _process_key_chain(val, allow_key_chain=allow_key_chain)
                if k in d:
                    assert isinstance(d[k], dict)
                    d[k] = _merge_dict(d[k], val)
                else:
                    d[k] = val
            return out

        if isinstance(opts, Config):
            opt_cfg_dict = opts._cfg_dict
        else:
            opt_cfg_dict = _process_key_chain(opts, allow_key_chain=allow_key_chain)
            opt_cfg_dict = ConfigDict(opt_cfg_dict)

        cfg_dict = self._cfg_dict
        object.__setattr__(self, '_cfg_dict', Config._merge_b_into_a(
            cfg_dict, opt_cfg_dict, allow_int_key=allow_int_key))

    @staticmethod
    def _merge_b_into_a(cfg_dict_a, cfg_dict_b, allow_int_key=False):
        if not (isinstance(cfg_dict_a, (dict, list, tuple) if allow_int_key else dict) and
                isinstance(cfg_dict_b, dict)):
            return cfg_dict_b

        cfg_dict_a = cfg_dict_a.copy()

        for key, val in cfg_dict_b.items():
            if allow_int_key and key.isdigit() and isinstance(cfg_dict_a, (list, tuple)):
                key = int(key)
                assert key < len(cfg_dict_a), f"Index {key} exceeds the length of {cfg_dict_a}"
                cfg_dict_a[key] = Config._merge_b_into_a(cfg_dict_a[key], val, allow_int_key=allow_int_key)

            else:
                if key in cfg_dict_a:
                    cfg_dict_a[key] = Config._merge_b_into_a(cfg_dict_a[key], val, allow_int_key=allow_int_key)
                else:
                    cfg_dict_a[key] = val
        return cfg_dict_a

    def set_from(self, key, dict_or_ctx, default=None):
        if key in dict_or_ctx:
            self[key] = dict_or_ctx[key]
        self.setdefault(key, default)
        return self[key]

    def save(self, out_dir, file_name=None, timestamp=None):
        out_name = 'config'
        if file_name:
            out_name = f'{out_name}_{osp.splitext(osp.basename(file_name))[0]}'
        if timestamp:
            out_name = f'{out_name}_{timestamp}'

        out_path = osp.join(out_dir, f'{out_name}.yaml')
        dump_yaml(self._cfg_dict.to_dict(), out_path)
        return out_path
