import os.path as osp

import torch

from ..config import Config
from .. import utils as U
from .. import registry as R
from .. import data as D
from .. import optim as O
from .. import plugins as P
from .runners import ComplexTrainRunner


@ComplexTrainRunner.register_hook('train_begin')
def prepare_setup(ctx):
    """ initial ctx contains:
        - config
        - options (will be merged into config)
        - work_dir
        - timestamp
    """
    # load config
    assert ctx.is_not('config', None)
    if isinstance(ctx.config, str):
        ctx.config_file = ctx.config
        ctx.config = Config.from_file(ctx.config)
        print(f"Loaded config file: {ctx.config_file}")
    config = ctx.config

    # merge options
    assert ctx.isinstance('options', dict)
    config.merge(ctx.options)

    # work dir
    assert ctx.has('work_dir')
    ctx.work_dir = U.abspath(ctx.work_dir)
    U.mkdir(ctx.work_dir, exist_ok=True, exist_rm=ctx.get('clear_work_dir', False))

    # logger
    assert ctx.has('timestamp')
    ctx.log_file = osp.join(ctx.work_dir, f'{ctx.timestamp}.log')
    ctx.logger = U.get_root_logger(log_file=ctx.log_file, log_level=config.log_level)

    # env info
    ctx.env_info = U.collect_env(pretty_print=True)
    ctx.logger.info('Environment info:\n' + U.DASH_LINE + ctx.env_info + '\n' + U.DASH_LINE)

    # device (gpu)
    ctx.gpu_id = ctx.options.gpu_id
    assert isinstance(ctx.gpu_id, int) and ctx.gpu_id >= 0
    assert torch.cuda.is_available()
    ctx.device = torch.device('cuda', ctx.gpu_id)

    # cudnn
    if config.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    # seed
    if config.seed is not None:
        U.set_random_seed(config.seed, config.deterministic)
        ctx.logger.info(f"Set random seed to {config.seed}, deterministic: {config.deterministic}")

    # save config
    path = config.save(ctx.work_dir, ctx.get('config_file'), ctx.get('timestamp'))
    ctx.logger.info(f"Saved a copy of current config to {path}")


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_setup')
def prepare_run_scheduler(ctx):
    # run scheduler
    run_scheduler_cfg = {**ComplexTrainRunner.run_scheduler_cfg, **ctx.config.complex_train}
    ctx.run_scheduler = R.RUN_SCHEDULERS.create(run_scheduler_cfg)
    ctx.logger.info(f"Created run scheduler {run_scheduler_cfg['type']}")


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_run_scheduler')
def prepare_model(ctx):
    # model
    ctx.model = R.MODELS.create(ctx.config.model).to(ctx.device)
    ctx.dtype = ctx.model.dtype
    ctx.logger.info(f"Built model {ctx.config.model.type}")


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_model')
def prepare_threads_data(ctx):
    data_cfg = ctx.config.data
    data_cfg.update_by_common(('data_source', 'data_params'))

    for thread in ctx.run_scheduler.get_threads():
        thread_ctx = thread.ctx

        # data
        if thread_ctx.has('data'):
            data_key = thread_ctx.data
            dataset_cfg = data_cfg[data_key].to_dict()
            dataloader_cfg = dataset_cfg.pop('data_loader', {})
            thread_ctx.dataset = R.DATASETS.create(dataset_cfg)
            thread_ctx.dataloader = D.get_dataloader(thread_ctx.dataset, dataloader_cfg)
            ctx.logger.info(f"[Thread {thread.name}] Created dataloader of {thread_ctx.dataset.name} "
                            f"(num_workers: {thread_ctx.dataloader.num_workers})")

        # auxiliary data
        if thread_ctx.has('auxiliary_data'):
            aux_dataset_cfgs = []
            for data_key in U.tolist(thread_ctx.auxiliary_data):
                aux_dataset_cfgs.append(data_cfg[data_key].to_dict())
            if aux_dataset_cfgs:
                thread_ctx.aux_datasets = []
                thread_ctx.aux_dataloaders = []
                for i, aux_dataset_cfg in enumerate(aux_dataset_cfgs):
                    aux_dataloader_cfg = aux_dataset_cfg.pop('data_loader', {})
                    aux_dataset = R.DATASETS.create(aux_dataset_cfg)
                    aux_dataloader = D.get_infinite_dataloader(aux_dataset, aux_dataloader_cfg)
                    thread_ctx.aux_datasets.append(aux_dataset)
                    thread_ctx.aux_dataloaders.append(aux_dataloader)
                    ctx.logger.info(f"[Thread {thread.name}] Create auxiliary dataloader of {thread_ctx.dataset.name} "
                                    f"(num_workers: {aux_dataloader.num_workers})")


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_threads_data')
def prepare_threads_iter(ctx):
    for thread in ctx.run_scheduler.get_threads():
        thread_ctx = thread.ctx

        # data iter
        if thread_ctx.has('dataloader'):
            thread_ctx.data_size = thread_ctx.dataset.size
            thread_ctx.data_iter = D.get_data_iter(thread_ctx.dataloader)
            thread_ctx.epoch = thread_ctx.get('start_epoch', 0)
            thread_ctx.inner_epoch = 0
            thread_ctx.step = thread_ctx.get('start_step', 0)
            thread_ctx.inner_step = 0
            thread_ctx.step_in_epoch = 0
            thread_ctx.steps_per_epoch = thread_ctx.data_iter.epoch_steps

        if thread_ctx.has('aux_dataloaders'):
            thread_ctx.aux_data_iters = [D.get_data_iter(dataloader)
                                         for dataloader in thread_ctx.aux_dataloaders]


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_threads_iter')
def prepare_threads_optim(ctx):
    for thread in ctx.run_scheduler.get_threads():
        thread_ctx = thread.ctx

        if thread_ctx.has('optim'):
            optim_cfg = thread_ctx.optim
            thread_ctx.mode = 'train'
            thread_ctx.model = ctx.model

            # optimizer
            module_key = optim_cfg.get('module')
            optim_module = [ctx.model.get_submodule(mod_key) for mod_key in U.tolist(module_key)] if module_key else ctx.model
            thread_ctx.optimizer = O.create_optimizer(optim_cfg.optimizer, optim_module)
            ctx.logger.info(f"[Thread {thread.name}] Created optimizer {optim_cfg.optimizer.type}")

            # lr scheduler
            if optim_cfg.has('lr_scheduler'):
                thread_ctx.lr_scheduler = R.LR_SCHEDULERS.create(optim_cfg.lr_scheduler)
                ctx.logger.info(f"[Thread {thread.name}] Created lr scheduler {optim_cfg.lr_scheduler.type}")
                thread_ctx.lr_scheduler.call_before_run(thread_ctx)

            # momentum scheduler
            if optim_cfg.has('momentum_scheduler'):
                thread_ctx.momentum_scheduler = R.MOMENTUM_SCHEDULERS.create(optim_cfg.momentum_scheduler)
                ctx.logger.info(f"[Thread {thread.name}] Created momentum scheduler {optim_cfg.momentum_scheduler.type}")
                thread_ctx.momentum_scheduler.call_before_run(thread_ctx)

            # optim scheduler
            thread_ctx.optim_scheduler = R.OPTIMIZE_SCHEDULERS.create(optim_cfg.optim_scheduler)
            ctx.logger.info(f"[Thread {thread.name}] Created optimize scheduler {optim_cfg.optim_scheduler.type}")
            thread_ctx.optim_scheduler.call_before_run(thread_ctx)


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_threads_optim')
def prepare_threads_plugins(ctx):
    for thread in ctx.run_scheduler.get_threads():
        thread_ctx = thread.ctx

        # train mode
        if thread_ctx.eq('mode', 'train'):
            # summarizer
            thread_ctx.summarizer = P.ThreadSummarizer(thread.name, **thread_ctx.summarize)
            ctx.logger.info(f"[Thread {thread.name}] Created summarizer")
            thread_ctx.summarizer.call_before_run(thread_ctx, ctx)

        # val mode
        if thread_ctx.eq('mode', 'val'):
            # summarizer
            progressbar = U.get_progressbar(thread_ctx.progressbar_unit, thread_ctx)
            thread_ctx.summarize.progressbar = progressbar
            thread_ctx.summarizer = P.ThreadSummarizer(thread.name, **thread_ctx.summarize)
            ctx.logger.info(f"[Thread {thread.name}] Created summarizer")
            thread_ctx.summarizer.call_before_run(thread_ctx, ctx)

            # validator
            thread_ctx.validator = P.ThreadValidator(thread.name, **thread_ctx.validate)
            ctx.logger.info(f"[Thread {thread.name}] Created validator")
            thread_ctx.validator.call_before_run(thread_ctx, ctx)

        # any mode
        # saver
        if thread_ctx.has('save'):
            thread_ctx.saver = P.ThreadSaver(thread.name, **thread_ctx.save)
            ctx.logger.info(f"[Thread {thread.name}] Created saver")
            thread_ctx.saver.call_before_run(thread_ctx, ctx)


@ComplexTrainRunner.register_hook('train_begin', dependency='prepare_threads_plugins')
def start_train(ctx):
    pass


@ComplexTrainRunner.register_hook('enter_station')
def station_reset_threads(ctx):
    station = ctx.station
    for thread in station.get_threads():
        thread_ctx = thread.ctx

        if thread_ctx.has('inner_epoch'):
            thread_ctx.inner_epoch = 0
        if thread_ctx.has('inner_step'):
            thread_ctx.inner_step = 0


@ComplexTrainRunner.register_hook('enter_thread')
def thread_prepare_epoch(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # epoch begin
    if thread_ctx.has('data_iter') and thread_ctx.data_iter.epoch_begin:
        if thread_ctx.has('step_in_epoch'):
            thread_ctx.step_in_epoch = 0

        # train mode
        if thread_ctx.eq('mode', 'train'):
            if thread_ctx.has('lr_scheduler'):
                thread_ctx.lr_scheduler.call_before_epoch(thread_ctx)
            if thread_ctx.has('momentum_scheduler'):
                thread_ctx.momentum_scheduler.call_before_epoch(thread_ctx)

        # val mode
        if thread_ctx.eq('mode', 'val'):
            thread_ctx.step = 0
            thread_ctx.validator.call_before_epoch(thread_ctx, ctx)

        # any mode
        if thread_ctx.has('summarizer'):
            thread_ctx.summarizer.call_before_epoch(thread_ctx, ctx)


@ComplexTrainRunner.register_hook('enter_thread', dependency='thread_prepare_epoch')
def thread_prepare_data(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    if thread_ctx.has('data_iter'):
        thread_ctx.batch = thread_ctx.data_iter.next()
        thread_ctx.batch_size = thread_ctx.batch['batch_size']

    if thread_ctx.has('aux_data_iters'):
        thread_ctx.aux_batches = [data_iter.next() for data_iter in thread_ctx.aux_data_iters]
        thread_ctx.aux_batch_sizes = [aux_batch['batch_size'] for aux_batch in thread_ctx.aux_batches]


@ComplexTrainRunner.register_hook('enter_thread', dependency='thread_prepare_data')
def thread_prepare_step(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # step begin
    # train mode
    if thread_ctx.eq('mode', 'train'):
        ctx.model.train()
        if thread_ctx.has('lr_scheduler'):
            thread_ctx.lr_scheduler.call_before_step(thread_ctx)
        if thread_ctx.has('momentum_scheduler'):
            thread_ctx.momentum_scheduler.call_before_step(thread_ctx)

    # val mode
    elif thread_ctx.eq('mode', 'val'):
        ctx.model.eval()

    # any mode
    if thread_ctx.has('summarizer'):
        thread_ctx.summarizer.call_before_step(thread_ctx, ctx)


@ComplexTrainRunner.register_hook('execute_thread')
def thread_do_train_step(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # train mode
    if thread_ctx.eq('mode', 'train'):
        batch = U.to(thread_ctx.batch, ctx.device, ctx.dtype)
        results = ctx.model.train_step(batch['inputs'], batch['targets'], batch_dict=batch, **thread_ctx.train_step_kwargs)
        thread_ctx.loss = results['loss']
        thread_ctx.results = results

        if not thread_ctx.has('aux_batches'):
            thread_ctx.optim_scheduler.call_at_step(thread_ctx)
        else:
            n_callpoints = 1 + len(thread_ctx.aux_batches)
            callpoint = 0
            thread_ctx.optim_scheduler.call_at_step(thread_ctx, callstage=(callpoint, n_callpoints))
            callpoint += 1

            for aux_batch in thread_ctx.aux_batches:
                aux_batch = U.to(aux_batch, ctx.device, ctx.dtype)
                results = ctx.model.train_step(aux_batch['inputs'], aux_batch['targets'], batch_dict=aux_batch, **thread_ctx.train_step_aux_kwargs)
                thread_ctx.loss = results['loss']
                thread_ctx.results = ctx.model.merge_results(thread_ctx.results, results)

                thread_ctx.optim_scheduler.call_at_step(thread_ctx, callstage=(callpoint, n_callpoints))
                callpoint += 1


@ComplexTrainRunner.register_hook('execute_thread')
def thread_do_val_step(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # val mode
    if thread_ctx.eq('mode', 'val'):
        with torch.no_grad():
            batch = U.to(thread_ctx.batch, ctx.device, ctx.dtype)
            results = ctx.model.eval_step(batch['inputs'], batch, **thread_ctx.val_step_kwargs)
            thread_ctx.results = results


@ComplexTrainRunner.register_hook('exit_thread')
def thread_cleanup_step(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # step end
    if thread_ctx.has('data_iter'):
        if thread_ctx.has('summarizer'):
            thread_ctx.summarizer.call_after_step(thread_ctx, ctx)

        if thread_ctx.has('saver'):
            thread_ctx.saver.call_after_step(thread_ctx, ctx)

        # val mode
        if thread_ctx.eq('mode', 'val'):
            thread_ctx.validator.call_after_step(thread_ctx, ctx)

        thread_ctx.step += 1
        thread_ctx.inner_step += 1
        thread_ctx.step_in_epoch += 1


@ComplexTrainRunner.register_hook('exit_thread', dependency='thread_cleanup_step')
def thread_cleanup_epoch(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    # epoch end
    if thread_ctx.has('data_iter') and thread_ctx.data_iter.epoch_end:
        if thread_ctx.has('summarizer'):
            thread_ctx.summarizer.call_after_epoch(thread_ctx, ctx)

        if thread_ctx.has('saver'):
            thread_ctx.saver.call_after_epoch(thread_ctx, ctx)

        # val mode
        if thread_ctx.eq('mode', 'val'):
            thread_ctx.validator.call_after_epoch(thread_ctx, ctx)

        thread_ctx.epoch += 1
        thread_ctx.inner_epoch += 1


@ComplexTrainRunner.register_hook('exit_thread', dependency='thread_cleanup_epoch')
def thread_update_state(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    if thread_ctx.has('data_iter'):
        if thread_ctx.le('max_inner_epochs', thread_ctx.inner_epoch):
            thread.close()
        if thread_ctx.le('max_inner_steps', thread_ctx.inner_step):
            thread.close()
        if thread_ctx.le('max_epochs', thread_ctx.epoch):
            thread.finish()
        if thread_ctx.le('max_steps', thread_ctx.step):
            thread.finish()


@ComplexTrainRunner.register_hook('exit_thread', dependency='thread_update_state')
def thread_update_global(ctx):
    thread = ctx.thread
    thread_ctx = thread.ctx

    if thread_ctx.has('global_vars'):
        for var_name in U.tolist(thread_ctx.global_vars):
            if thread_ctx.has(var_name):
                ctx[var_name] = thread_ctx[var_name]


@ComplexTrainRunner.register_hook('exit_station')
def station_cleanup_threads(ctx):
    pass


@ComplexTrainRunner.register_hook('train_end')
def finish_train(ctx):
    pass
