import os.path as osp

import torch

from ..config import Config
from .. import utils as U
from .. import registry as R
from .. import data as D
from .. import optim as O
from .. import plugins as P
from .runners import TrainRunner


@TrainRunner.register_hook('train_begin')
def prepare_setup(ctx):
    """ initial ctx contains:
        - config
        - options (will be merged into config)
        - work_dir
        - timestamp 
    """
    # load config
    assert ctx.has('config') and ctx.config is not None
    if isinstance(ctx.config, str):
        ctx.config_file = ctx.config
        ctx.config = Config.from_file(ctx.config)
        print(f"Loaded config file: {ctx.config_file}")
    config = ctx.config

    # merge options
    assert ctx.has('options') and isinstance(ctx.options, dict)
    config.merge(ctx.options)

    # work dir
    assert ctx.has('work_dir')
    ctx.work_dir = U.abspath(ctx.work_dir)
    U.mkdir(ctx.work_dir, exist_ok=True)

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


@TrainRunner.register_hook('train_begin', dependency='prepare_setup')
def prepare_model(ctx):
    config = ctx.config
    ctx.model = R.MODELS.create(config.model).to(ctx.device)
    ctx.dtype = ctx.model.dtype
    ctx.logger.info(f"Built model {config.model.type}")

    # ctx.model_info = ctx.model.info(input_size=config.data.data_params.input_size, pretty_print=True)
    # ctx.logger.info('Model info:\n' + U.DASH_LINE + ctx.model_info + '\n' + U.DASH_LINE)


@TrainRunner.register_hook('train_begin', dependency='prepare_model')
def prepare_data(ctx):
    data_cfg = ctx.config.data
    common_source = data_cfg.get('data_source', {})
    common_params = data_cfg.get('data_params', {})

    data_cfg.train.update_at_key('data_source', common_source, overwrite=False)
    data_cfg.train.update_at_key('data_params', common_params, overwrite=False)
    train_dataset_cfg = data_cfg.train.to_dict()
    train_dataloader_cfg = train_dataset_cfg.pop('data_loader', {})
    auxiliary_data = train_dataset_cfg.pop('auxiliary_data', None)
    ctx.train_dataset = R.DATASETS.create(train_dataset_cfg)
    ctx.train_dataloader = D.get_dataloader(ctx.train_dataset, train_dataloader_cfg)

    ctx.logger.info(f"Train dataloader: batch_size = {train_dataloader_cfg['batch_size']}, "
                    f"num_workers = {train_dataloader_cfg['num_workers']}")

    if auxiliary_data:
        aux_dataset_cfgs = []
        for key in U.tolist(auxiliary_data):
            if not data_cfg.has(key):
                continue
            data_cfg[key].update_at_key('data_params', common_params, overwrite=False)
            aux_dataset_cfgs.append(data_cfg[key].to_dict())

        if aux_dataset_cfgs:
            ctx.aux_datasets = []
            ctx.aux_dataloaders = []
            for i, aux_dataset_cfg in enumerate(aux_dataset_cfgs):
                aux_dataloader_cfg = aux_dataset_cfg.pop('data_loader', {})
                ctx.aux_datasets.append(R.DATASETS.create(aux_dataset_cfg))
                ctx.aux_dataloaders.append(D.get_infinite_dataloader(ctx.aux_datasets[-1], aux_dataloader_cfg))

                ctx.logger.info(f"Aux dataloader {i}: batch_size = {aux_dataloader_cfg['batch_size']}, "
                                f"num_workers = {aux_dataloader_cfg['num_workers']}")

    if ctx.config.validate:
        data_cfg.val.update_at_key('data_source', common_source, overwrite=False)
        if common_source and common_source.has('drop_prob'):
            data_cfg.val.data_source.drop_imgs = ctx.train_dataset.image_ids
        data_cfg.val.update_at_key('data_params', common_params, overwrite=False)
        val_dataset_cfg = data_cfg.val.to_dict()
        val_dataloader_cfg = val_dataset_cfg.pop('data_loader', {})
        ctx.val_dataset = R.DATASETS.create(val_dataset_cfg)
        ctx.val_dataloader = D.get_dataloader(ctx.val_dataset, val_dataloader_cfg)

        ctx.logger.info(f"Val dataloader: batch_size = {val_dataloader_cfg['batch_size']}, "
                        f"num_workers = {val_dataloader_cfg['num_workers']}")


@TrainRunner.register_hook('train_begin', dependency='prepare_data')
def prepare_optim(ctx):
    optim_cfg = ctx.config.optim

    ctx.max_epochs = optim_cfg.max_epochs
    ctx.steps_per_epoch = len(ctx.train_dataloader)
    ctx.max_steps = optim_cfg.max_epochs * ctx.steps_per_epoch

    ctx.train_step = 0
    ctx.optimizer = O.create_optimizer(optim_cfg.optimizer, ctx.model)
    ctx.logger.info(f'Created optimizer {optim_cfg.optimizer.type}')

    if optim_cfg.has('lr_scheduler'):
        ctx.lr_scheduler = R.LR_SCHEDULERS.create(optim_cfg.lr_scheduler)
        ctx.logger.info(f'Created lr scheduler {optim_cfg.lr_scheduler.type}')
        ctx.lr_scheduler.call_before_run(ctx)

    if optim_cfg.has('momentum_scheduler'):
        ctx.momentum_scheduler = R.MOMENTUM_SCHEDULERS.create(optim_cfg.momentum_scheduler)
        ctx.logger.info(f'Created momentum scheduler {optim_cfg.momentum_scheduler.type}')
        ctx.momentum_scheduler.call_before_run(ctx)

    ctx.optim_scheduler = R.OPTIMIZE_SCHEDULERS.create(optim_cfg.optim_scheduler)
    ctx.logger.info(f'Created optimize scheduler {optim_cfg.optim_scheduler.type}')
    ctx.optim_scheduler.call_before_run(ctx)


@TrainRunner.register_hook('train_begin', dependency='prepare_optim')
def prepare_plugins(ctx):
    config = ctx.config

    ctx.summarizer = P.Summarizer(**config.summarizer)
    ctx.validator = P.Validator(**config.validator)
    ctx.saver = P.Saver(**config.saver)
    ctx.timer = U.Timer()

    if config.is_('use_diagnosis', True):
        ctx.diagnoser = P.Diagnoser(**config.diagnoser)
        ctx.diagnoser.call_before_run(ctx)

    ctx.summarizer.call_before_run(ctx)
    ctx.validator.call_before_run(ctx)
    ctx.saver.call_before_run(ctx)


@TrainRunner.register_hook('train_begin', dependency='prepare_plugins')
def prepare_train(ctx):
    ctx.model.call_before_train(ctx)

    if ctx.has('aux_dataloaders'):
        ctx.aux_dataiterators = [iter(dataloader) for dataloader in ctx.aux_dataloaders]


@TrainRunner.register_hook('train_iter_epochs')
def iterate_train_epochs(ctx):
    for epoch in range(ctx.max_epochs):
        yield {'epoch': epoch}


@TrainRunner.register_hook('train_epoch_begin')
def prepare_train_epoch(ctx):
    ctx.model.train()
    ctx.mode = 'train'

    if ctx.has('lr_scheduler'):
        ctx.lr_scheduler.call_before_epoch(ctx)

    if ctx.has('momentum_scheduler'):
        ctx.momentum_scheduler.call_before_epoch(ctx)

    ctx.summarizer.call_before_epoch(ctx)


@TrainRunner.register_hook('train_iter_steps')
def iterate_train_steps(ctx):
    for i, train_batch in enumerate(ctx.train_dataloader):
        batch_info = {
            'train_inner_step': i,
            'train_batch': train_batch,
            'train_batch_size': train_batch['batch_size'],
            'train_loss': ctx.train_dataset.data_params.get('loss'),
            'train_weight': ctx.train_dataset.data_params.get('loss_weight', 1.)
        }

        if ctx.has('aux_dataiterators'):
            batch_info.update({
                'aux_batches': [],
                'aux_batch_sizes': [],
                'aux_losses': [],
                'aux_weights': []
            })
            for dataset, dataiterator in zip(ctx.aux_datasets, ctx.aux_dataiterators):
                aux_batch = next(dataiterator)
                aux_batch_size = aux_batch['batch_size']
                aux_loss = dataset.data_params.get('loss')
                aux_weight = dataset.data_params.get('loss_weight', 1.)
                batch_info['aux_batches'].append(aux_batch)
                batch_info['aux_batch_sizes'].append(aux_batch_size)
                batch_info['aux_losses'].append(aux_loss)
                batch_info['aux_weights'].append(aux_weight)

        yield batch_info


@TrainRunner.register_hook('train_step_begin')
def prepare_train_step(ctx):
    if ctx.has('lr_scheduler'):
        ctx.lr_scheduler.call_before_step(ctx)

    if ctx.has('momentum_scheduler'):
        ctx.momentum_scheduler.call_before_step(ctx)

    ctx.summarizer.call_before_step(ctx)

    if ctx.has('diagnoser'):
        ctx.diagnoser.call_before_step(ctx)


@TrainRunner.register_hook('train_step')
def do_train_step(ctx):
    train_batch = ctx.train_batch
    train_batch['inputs'] = U.to(train_batch['inputs'], ctx.device, ctx.dtype)
    train_batch['targets'] = U.to(train_batch['targets'], ctx.device, ctx.dtype)
    train_loss = ctx.train_loss
    train_weight = ctx.train_weight

    results = ctx.model.train_step(train_batch['inputs'], train_batch['targets'], batch_dict=train_batch, target_loss=train_loss)
    ctx.loss = results['loss'] * train_weight
    ctx.train_results = results

    if not ctx.has('aux_batches'):
        ctx.optim_scheduler.call_at_step(ctx)
    else:
        aux_batches = ctx.aux_batches
        aux_losses = ctx.aux_losses
        aux_weights = ctx.aux_weights

        n_callpoints = 1 + len(aux_batches)
        callpoint = 0
        ctx.optim_scheduler.call_at_step(ctx, callstage=(callpoint, n_callpoints))
        callpoint += 1

        for aux_batch, aux_loss, aux_weight in zip(aux_batches, aux_losses, aux_weights):
            aux_batch['inputs'] = U.to(aux_batch['inputs'], ctx.device, ctx.dtype)
            aux_batch['targets'] = U.to(aux_batch['targets'], ctx.device, ctx.dtype)

            res = ctx.model.train_step(aux_batch['inputs'], aux_batch['targets'], batch_dict=aux_batch, target_loss=aux_loss)
            ctx.loss = res['loss'] * aux_weight
            ctx.train_results = ctx.model.merge_results(ctx.train_results, res)

            ctx.optim_scheduler.call_at_step(ctx, callstage=(callpoint, n_callpoints))
            callpoint += 1


@TrainRunner.register_hook('train_step_end')
def finish_train_step(ctx):
    ctx.summarizer.call_after_step(ctx)
    ctx.validator.call_after_step(ctx)  # call val runner
    ctx.saver.call_after_step(ctx)

    if ctx.has('diagnoser'):
        ctx.diagnoser.call_after_step(ctx)

    ctx.train_step += 1


@TrainRunner.register_hook('train_epoch_end')
def finish_train_epoch(ctx):
    ctx.saver.call_after_epoch(ctx)
    ctx.validator.call_after_epoch(ctx)


@TrainRunner.register_hook('train_end')
def finish_train(ctx):
    ctx.model.call_after_train(ctx)
