import os.path as osp


def parse_path(path, is_file=None, is_dir=None):
    if path is None:
        return []

    if isinstance(path, str):
        path = path.strip(',')
        if ',' in path:
            path = path.split(',')
        else:
            path = [path]
    assert isinstance(path, (list, tuple))

    path_out = []
    for p in path:
        assert osp.exists(p), f"{p} not found"
        if is_file:
            assert osp.isfile(p), f"{p} not a file"
        if is_dir:
            assert osp.isdir(p), f"{p} not a directory"
        path_out.append(osp.realpath(p))
    return path_out


def parse_list(ls, typ=str):
    if isinstance(ls, str):
        ls = ls.strip(',')
        if ',' in ls:
            ls = ls.split(',')
        else:
            ls = [ls]
    assert isinstance(ls, (list, tuple))
    ls = [typ(e) for e in ls]
    return ls
