import os.path as osp

from myzonecv.core import Context
from myzonecv.core.registry import RUNNERS
from myzonecv.core.utils import get_timestamp, mkdir


def infer(config,
          runner='infer',
          data_infer=None,
          work_dir=None,
          timestamp=None,
          dump_mode='all',  # 'all' or 'step'
          cache_step=False,
          custom_dirs={},
          custom_vars={},
          gpu_id=0,
          log_level='INFO',
          **kwargs):
    kwargs.update({
        'gpu_id': gpu_id,
        'log_level': log_level
    })

    if work_dir is None:
        work_dir = './work'
    work_dir = osp.realpath(work_dir)
    mkdir(work_dir, exist_ok=True)

    timestamp = (timestamp or get_timestamp())

    if custom_dirs:
        for name, dir_path in custom_dirs.items():
            if name.endswith('_exist_rm') or not dir_path:
                continue
            exist_rm = custom_dirs.get(f'{name}_exist_rm', False)
            mkdir(dir_path, exist_ok=True, exist_rm=exist_rm)

    ctx = Context(config=config,
                  options=kwargs,
                  data_infer=data_infer,
                  work_dir=work_dir,
                  timestamp=timestamp,
                  dump_mode=dump_mode,
                  cache_step=cache_step,
                  **custom_dirs,
                  **custom_vars)

    runner = RUNNERS.create({'type': runner})
    runner.run(ctx)


def unified_infer(detect_config=None,
                  pose_config=None,
                  runner='unified_infer',
                  work_dir=None,
                  timestamp=None,
                  gpu_id=0,
                  display=True,
                  log_level='INFO',
                  **kwargs):
    kwargs.update({
        'gpu_id': gpu_id,
        'display': display,
        'log_level': log_level
    })

    if work_dir is None:
        work_dir = './work'
    work_dir = osp.realpath(work_dir)
    mkdir(work_dir, exist_ok=True)

    timestamp = (timestamp or get_timestamp())

    ctx = Context(detect_config=detect_config,
                  pose_config=pose_config,
                  options=kwargs,
                  work_dir=work_dir,
                  timestamp=timestamp)

    runner = RUNNERS.create({'type': runner})
    runner.run(ctx)
