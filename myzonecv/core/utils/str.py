import numpy as np


def list2str(a, delimiter=',', reduce=False, tmpl='{}'):
    if isinstance(a, str):
        return tmpl.format(a)

    if isinstance(a, np.ndarray):
        a = a.flatten().tolist()

    if not isinstance(a, (list, tuple)):
        return tmpl.format(a)

    a = [tmpl.format(v) for v in a]

    if len(a) == 1:
        return a[0]
    elif len(a) > 1 and all([a[0] == a[i] for i in range(1, len(a))]):
        return a[0]

    return delimiter.join(a)
