import os
import os.path as osp
import shutil
import tempfile


def abspath(path):
    if not path:
        return path

    return osp.expanduser(osp.abspath(path))


def mkdir(path, exist_ok=False, exist_rm=False):
    if exist_rm and os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

    os.makedirs(path, exist_ok=exist_ok)


def mktempdir():
    return tempfile.mkdtemp()


def rmdir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)


def sibling_dir(name, sibling=None):
    parent_dir = None
    if sibling:
        if not isinstance(sibling, (list, tuple)):
            sibling = [sibling]
        for s in sibling:
            if s:
                parent_dir = osp.dirname(s)
                break
    if not parent_dir:
        parent_dir = '.'
    return osp.join(parent_dir, name)


def cp(src, dst):
    if osp.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def mv(src, dst):
    shutil.move(src, dst)
