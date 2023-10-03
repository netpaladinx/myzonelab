import re
import json


def load_log(log_file):
    log = []
    with open(log_file) as fin:
        for line in fin.readlines():
            log.append(json.loads(line.strip()))
    return log


def get_curve(log, metric, head_regex=r'(?:train|val)@(?:step|epoch)\[(\d+)(?:/\d+)?\]'):
    x, y = [], []
    for l in log:
        mat = re.match(head_regex, l['head'])
        if mat:
            x.append(int(mat.group(1)))
        else:
            x.append(0)
        y.append(l[metric])
    return x, y
