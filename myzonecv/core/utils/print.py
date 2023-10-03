from .misc import round_float


def print_progress(n, N, interval=100, max_intervals=None, interval_factor=10, msg_tmpl='{}/{}', prefix='', logger_func=None):
    if max_intervals is not None:
        n_intervals = int(N / interval + 0.5)
        while n_intervals > max_intervals:
            interval = interval * interval_factor
            n_intervals = int(N / interval + 0.5)

    if (n + 1) % interval == 0:
        msg = msg_tmpl.format(n + 1, N)
        if prefix:
            msg = prefix + msg

        if logger_func is None:
            print(msg)
        else:
            logger_func(msg)


def print_dict(dct, prefix=None):
    lines = []
    line = []
    for k, v in dct.items():
        if isinstance(v, dict):
            lines.append(print_dict(v, prefix=(prefix or '') + f'{k}-> '))
        else:
            v = round_float(v)
            line.append(f'{k}: {v:.5f}' if isinstance(v, float) else f'{k}: {v}')
    line = (prefix or '') + ', '.join(line)
    lines.append(line)
    return '\n'.join(lines)
