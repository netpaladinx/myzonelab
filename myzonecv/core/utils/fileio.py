import json
import yaml
import ast

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import numpy as np


def load_json(path):
    with open(path, 'r') as fin:
        return json.load(fin)


def dump_json(obj, path):
    with open(path, 'w') as fout:
        json.dump(obj, fout)


def load_yaml(path):
    with open(path, 'r') as fin:
        return yaml.load(fin, Loader=Loader)


def dump_yaml(obj, path):
    with open(path, 'w') as fout:
        yaml.dump(obj, fout, Dumper=Dumper)


def load_numpy(path):
    with open(path, 'rb') as fin:
        return np.load(fin)


def dump_numpy(obj, path):
    with open(path, 'wb') as fout:
        np.save(fout, obj)


def read_numpy(path):
    obj = None

    with open(path) as fin:
        for line in fin.readlines():
            sp = line.strip().split(':')
            for i, p in enumerate(sp):
                if ',' in p:
                    sp[i] = [[ast.literal_eval(e) for e in p.split(',')]]
                else:
                    sp[i] = [ast.literal_eval(p)]

            if obj is None:
                obj = sp
            else:
                for o, p in zip(obj, sp):
                    o += p
    obj = [np.array(o) for o in obj]
    if len(obj) == 1:
        return obj[0]
    else:
        return obj


def write_numpy(obj, path):
    if not isinstance(obj, (list, tuple)):
        obj = (obj,)

    with open(path, 'w') as fout:
        for t in zip(*obj):
            line = []
            for r in t:
                if r.ndim == 0:
                    line.append(str(r))
                elif r.ndim == 1:
                    line.append(','.join([str(e) for e in r]))
                else:
                    raise ValueError("Only 1D and 2D numpy arrays are supported")
            line = ':'.join(line)
            fout.write(line + '\n')
