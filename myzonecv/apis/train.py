import os.path as osp

from myzonecv.core import Context
from myzonecv.core.registry import RUNNERS
from myzonecv.core.utils import get_timestamp


def train(config,
          runner='train',
          work_dir=None,
          timestamp=None,  # for tagging log files
          validate=True,
          gpu_id=0,
          cudnn_benchmark=False,
          deterministic=False,
          seed=None,
          log_level='INFO',
          clear_work_dir=False,
          **kwargs):
    kwargs.update({
        'validate': validate,
        'gpu_id': gpu_id,
        'cudnn_benchmark': cudnn_benchmark,
        'deterministic': deterministic,
        'seed': seed,
        'log_level': log_level
    })

    if work_dir is None:
        work_dir = './work'
    work_dir = osp.realpath(work_dir)

    timestamp = (timestamp or get_timestamp())

    ctx = Context(config=config,
                  options=kwargs,
                  work_dir=work_dir,
                  timestamp=timestamp,
                  clear_work_dir=clear_work_dir)

    runner = RUNNERS.create({'type': runner})
    runner.run(ctx)


def complex_train(config, **kwargs):
    train(config, runner='complex_train', **kwargs)
