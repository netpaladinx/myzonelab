import os.path as osp

from myzonecv.core import Context
from myzonecv.core.registry import RUNNERS
from myzonecv.core.utils import get_timestamp, mkdir


def test(config,
         runner='test',
         data_test=None,
         work_dir=None,
         timestamp=None,
         summary_path=None,
         evaluate_mode='all',  # 'all' or 'step'
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

    if not summary_path:
        summary_path = osp.join(work_dir, f'summary_{timestamp}.json')
    if not summary_path.endswith('.json'):
        summary_path += '.json'

    if custom_dirs:
        for name, dir_path in custom_dirs.items():
            if name.endswith('_exist_rm') or not dir_path:
                continue
            exist_rm = custom_dirs.get(f'{name}_exist_rm', False)
            mkdir(dir_path, exist_ok=True, exist_rm=exist_rm)

    ctx = Context(config=config,
                  options=kwargs,
                  data_test=data_test,
                  work_dir=work_dir,
                  timestamp=timestamp,
                  summary_path=summary_path,
                  evaluate_mode=evaluate_mode,
                  cache_step=cache_step,
                  **custom_dirs,
                  **custom_vars)

    runner = RUNNERS.create({'type': runner})
    runner.run(ctx)
