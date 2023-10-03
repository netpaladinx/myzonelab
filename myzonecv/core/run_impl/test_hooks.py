import torch

from ..config import Config
from .. import utils as U
from .. import registry as R
from .. import data as D
from .runners import TestRunner


@TestRunner.register_hook('test_begin')
def prepare_setup(ctx):
    """ initial ctx contains:
        - config
        - options (will be merged into config)
        - work_dir
    """
    # load config
    assert ctx.is_not('config', None)
    if isinstance(ctx.config, str):
        ctx.config = Config.from_file(ctx.config)
    config = ctx.config

    # merge options
    assert ctx.isinstance('options', dict)
    config.merge(ctx.options)

    # work dir
    assert ctx.has('work_dir')
    ctx.work_dir = U.abspath(ctx.work_dir)
    U.mkdir(ctx.work_dir, exist_ok=True)

    # logger
    ctx.logger = U.get_root_logger(log_file=ctx.get('log_path'), log_level=ctx.options.log_level)

    # device (gpu)
    ctx.gpu_id = ctx.options.gpu_id
    assert isinstance(ctx.gpu_id, int) and ctx.gpu_id >= 0
    assert torch.cuda.is_available()
    ctx.device = torch.device('cuda', ctx.gpu_id)


@TestRunner.register_hook('test_begin', dependency='prepare_setup')
def prepare_model(ctx):
    # model
    ctx.model = R.MODELS.create(ctx.config.model).to(ctx.device)
    ctx.dtype = ctx.model.dtype
    ctx.logger.info(f"Built model {ctx.config.model.type}")


@TestRunner.register_hook('test_begin', dependency='prepare_model')
def prepare_data(ctx):
    data_cfg = ctx.config.data

    if ctx.has('data_test') and ctx.data_test:
        data_cfg.test = data_cfg[ctx.data_test]
    data_cfg.update_by_common(('data_source', 'data_params'), 'test')

    # data
    dataset_cfg = data_cfg.test.to_dict()
    dataloader_cfg = dataset_cfg.pop('data_loader', {})
    ctx.dataset = R.DATASETS.create(dataset_cfg)
    ctx.dataloader = D.get_dataloader(ctx.dataset, dataloader_cfg)
    ctx.logger.info(f"Created dataloader of {ctx.dataset.name}: "
                    f"batch_size = {ctx.dataloader.batch_size}, "
                    f"num_workers = {ctx.dataloader.num_workers}")


@TestRunner.register_hook('test_begin', dependency='prepare_data')
def prepare_test(ctx):
    ctx.model.eval()
    ctx.model.call_before_eval(ctx)

    ctx.total_steps = len(ctx.dataloader)
    ctx.all_results = []


@TestRunner.register_hook('test_iter_steps')
def iterate_test_step(ctx):
    ctx.progress_bar = U.ProgressBar(len(ctx.dataset), start=True)

    for i, batch in enumerate(ctx.dataloader):
        yield {'step': i,
               'batch': batch,
               'batch_size': batch['batch_size']}


@TestRunner.register_hook('test_step_begin')
def prepare_test_step(ctx):
    pass


@TestRunner.register_hook('test_step')
def do_test_step(ctx):
    with torch.no_grad():
        batch = U.to(ctx.batch, ctx.device, ctx.dtype)
        results = ctx.model.eval_step(batch['inputs'], batch)
        ctx.results = results

        if ctx.evaluate_mode == 'all':
            if not ctx.cache_step:
                ctx.all_results.append(results)
            else:
                ctx.dataset.cache_step(results, batch, ctx=ctx)
        elif ctx.evaluate_mode == 'step':
            ctx.dataset.evaluate_step(results, batch, ctx=ctx)

        for _ in range(ctx.batch_size):
            ctx.progress_bar.update()


@TestRunner.register_hook('test_step_end')
def finish_test_step(ctx):
    pass


@TestRunner.register_hook('test_end')
def finish_test(ctx):
    ctx.progress_bar.end()

    if ctx.evaluate_mode == 'all':
        results = ctx.dataset.evaluate_all(ctx.all_results, ctx=ctx)
        if results:
            ctx.logger.info('Evaluation results: \n' + U.print_dict(results))

    ctx.dataset.summarize()
    ctx.model.call_after_eval(ctx)
