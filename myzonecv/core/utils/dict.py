# Based on https://github.com/mewwts/addict
import copy

from .misc import hashable, iterable


class Dict(dict):
    def __init__(self, *args, **kwargs):
        # '__parent' and '__key' will be deleted after setting the first key/value
        object.__setattr__(self, '__parent', kwargs.pop('__parent', None))
        object.__setattr__(self, '__key', kwargs.pop('__key', None))

        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):  # {key: value, ...}
                for key, val in arg.items():
                    self[key] = self.copy_val(val)
            elif isinstance(arg, (list, tuple)) and len(arg) == 2 and hashable(arg[0]):  # (key, value)
                self[arg[0]] = self.copy_val(arg[1])
            elif iterable(arg):  # (key, value), ...
                for a in iter(arg):
                    if isinstance(a, (list, tuple)) and len(a) == 2 and hashable(a[0]):
                        self[a[0]] = self.copy_val(a[1])

        for key, val in kwargs.items():
            self[key] = self.copy_val(val)

    def copy_val(self, val):  # recursion only happens in __init__()
        cls = self.__class__
        if isinstance(val, dict) and not isinstance(val, cls):
            return cls(val)
        elif isinstance(val, (list, tuple)):
            return type(val)(self.copy_val(elem) for elem in val)
        return val

    def __setitem__(self, key, val):
        super().__setitem__(key, val)

        try:
            __parent = object.__getattribute__(self, '__parent')
            __key = object.__getattribute__(self, '__key')

            if __parent is not None:
                __parent[__key] = self  # add self to its parent

            object.__delattr__(self, '__parent')
            object.__delattr__(self, '__key')
        except AttributeError:
            pass

    def __setattr__(self, name, val):
        if hasattr(self.__class__, name):
            raise AttributeError(f"Dict's object attribute '{name}' is read-only")
        else:
            self[name] = val

    def __getattr__(self, name):
        return self[name]

    def __missing__(self, key):  # called by __getitem__ for any missing key
        """ Allows a missing key with a new Dict instance returned """
        return self.__class__(__parent=self, __key=key)

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        dct = {}
        for key, val in self.items():
            if isinstance(val, self.__class__):
                dct[key] = val.to_dict()
            elif isinstance(val, (list, tuple)):
                dct[key] = type(val)(elem.to_dict() if isinstance(elem, self.__class__) else elem
                                     for elem in val)
            else:
                dct[key] = val
        return dct

    def copy(self):
        return copy.copy(self)

    def deepcopy(self):
        return copy.deepcopy(self)

    def __deepcopy__(self, memo):
        other = self.__class__()
        memo[id(self)] = other
        for key, val in self.items():
            other[copy.deepcopy(key, memo)] = copy.deepcopy(val, memo)
        return other

    def update(self, *args, overwrite=True, **kwargs):  # no recursion!
        if args:
            assert len(args) == 1 and isinstance(args[0], dict)
            in_dict = args[0]
        else:
            in_dict = kwargs
        for key, val in in_dict.items():
            if key not in self or overwrite:
                self[key] = val

    def __getnewargs__(self):
        return tuple(self.items())

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        else:
            self[key] = default
            return default

    def has(self, *names):
        return all(name in self.keys() for name in names)

    def eq(self, key, val):
        return self.has(key) and self[key] == val

    def ne(self, key, val):
        return self.has(key) and self[key] != val

    def gt(self, key, val):
        return self.has(key) and self[key] > val

    def lt(self, key, val):
        return self.has(key) and self[key] < val

    def ge(self, key, val):
        return self.has(key) and self[key] >= val

    def le(self, key, val):
        return self.has(key) and self[key] <= val

    def is_(self, key, val):
        return self.has(key) and self[key] is val

    def is_not(self, key, val):
        return self.has(key) and self[key] is not val

    def isinstance(self, key, type_):
        return self.has(key) and isinstance(self[key], type_)

    def update_at_key(self, key, *args, overwrite=True, **kwargs):
        self.setdefault(key, self.__class__(__parent=self, __key=key)).update(*args, overwrite=overwrite, **kwargs)

    def update_by_common(self, common_key, update_key=None, overwrite=False):
        if not isinstance(common_key, (list, tuple)):
            common_key = (common_key,)
        if update_key and not isinstance(update_key, (list, tuple)):
            update_key = (update_key,)

        for key, val in self.items():
            if key in common_key:
                continue
            if update_key and key not in update_key:
                continue
            for k in common_key:
                c = self.get(k, {})
                if c:
                    val.update_at_key(k, c, overwrite=overwrite)


def get_if_is(dct, key, val, default):
    if val is default and dct is not None:
        return dct.get(key, default)
    return val


def get_if_eq(dct, key, val, default):
    if val == default and dct is not None:
        return dct.get(key, default)
    return val
