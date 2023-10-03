import os
import os.path as osp
import glob

from myzonecv.core.errors import ConfigFileNotFound

CONFIG_DESCRIPTION = """
{task_type}_{network_type}_{network_instance_tags:[network_size|train_dataset|input_size|other_settings]+}
"""


def list_config_dirs():
    conf_dir = osp.dirname(osp.realpath(__file__))
    exp_dir = osp.join(conf_dir, '../experimental')

    conf_dirs = [conf_dir]
    try:
        import myzonecv.experimental
        for path in os.listdir(exp_dir):
            path = osp.join(exp_dir, path)
            if osp.isdir(path) and osp.isdir(osp.join(path, 'configs')):
                conf_dirs.append(osp.realpath(osp.join(path, 'configs')))
    except Exception:
        pass
    return conf_dirs


def get_config_file(file_name):
    if not file_name.endswith('.yaml'):
        file_name = file_name + '.yaml'

    conf_dirs = [''] + list_config_dirs()
    for conf_dir in conf_dirs:
        file_path = osp.join(conf_dir, file_name) if conf_dir else file_name
        if osp.exists(file_path):
            return osp.realpath(file_path)

    print(f'config name format: {CONFIG_DESCRIPTION}')
    print(f'current config list: {list_configs()}')
    raise ConfigFileNotFound(f"File {file_path} not found")


def list_configs():
    confs = []
    for conf_dir in list_config_dirs():
        for conf_path in glob.glob(osp.join(conf_dir, '*.yaml')):
            confs.append(osp.basename(conf_path).rsplit('.', 1)[0])
    return confs


__all__ = ['list_config_dirs', 'list_configs', 'get_config_file']
