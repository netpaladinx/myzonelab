import time
from itertools import repeat
from collections import OrderedDict, abc

import numpy as np


def hashable(a):
    try:
        hash(a)
    except TypeError:
        return False
    return True


def iterable(a):
    try:
        iter(a)
    except TypeError:
        return False
    return True


def str2int(a):
    try:
        return int(a)
    except ValueError:
        return None


def get_timestamp(t=None, fmt='%Y%m%d_%H%M%S'):
    if not t:
        t = time.localtime()
    return time.strftime(fmt, t)


def round_float(a, ndigits=5, tmpl=None):
    if isinstance(a, (tuple, list)):
        return type(a)(round_float(i, tmpl=tmpl) for i in a)
    elif isinstance(a, dict):
        return OrderedDict([(k, round_float(v, tmpl=tmpl)) for k, v in a.items()])
    elif isinstance(a, (float, np.float32, np.float64, np.float16)):
        val = round(float(a), ndigits)
        if tmpl:
            val = tmpl.format(val)
        return val
    else:
        return a


def tolist(a, exclude=(), treat_none=True):
    if a is None:
        if treat_none:
            a = []
        else:
            return None
    elif isinstance(a, list):
        pass
    elif isinstance(a, tuple):
        a = list(a)
    else:
        a = [a]

    if not exclude:
        return a
    elif isinstance(exclude, (list, tuple)):
        return [i for i in a if i not in exclude]
    else:
        return [i for i in a if i != exclude]


def partial(func, /, *args, **keywords):
    def newfunc(*fargs, **fkeywords):
        newkeywords = {**keywords, **fkeywords}
        return func(*args, *fargs, **newkeywords)
    newfunc.func = func
    newfunc.args = args
    newfunc.keywords = keywords
    return newfunc


def ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, abc.Iterable):
            x = tuple(x)
            assert len(x) == n
            return x
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


single = ntuple(1, "single")
pair = ntuple(2, "pair")
triple = ntuple(3, "triple")
quadruple = ntuple(4, "quadruple")
