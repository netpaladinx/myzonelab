import cv2
import torch

from ..config import Config
from .. import utils as U
from .. import registry as R
from .. import data as D
from .runners import UnifiedInferRunner


@UnifiedInferRunner.register_hook('infer_begin')
def prepare_setup(ctx):
    """ initial ctx contains:
        - detect_config
        - pose_config
        - options
        - work_dir
    """
    # detect config and pose config
    assert ctx.has('detect_config', 'pose_config', 'options')
    detect_opts, pose_opts = U.collect_options(ctx.options, ['detect.', 'pose.'])

    assert ctx.detect_config is not None
    if isinstance(ctx.detect_config, str):
        ctx.detect_config = Config.from_file(ctx.detect_config)
    ctx.detect_config.merge(detect_opts)

    if ctx.pose_config is not None:
        if isinstance(ctx.pose_config, str):
            ctx.pose_config = Config.from_file(ctx.pose_config)
        ctx.pose_config.merge(pose_opts)

    # work dir
    assert ctx.has('work_dir')
    ctx.work_dir = U.abspath(ctx.work_dir)
    U.mkdir(ctx.work_dir, exist_ok=True)

    # logger
    ctx.logger = U.get_root_logger(log_level=ctx.options.log_level)

    # device (gpu)
    ctx.gpu_id = ctx.options.gpu_id
    assert isinstance(ctx.gpu_id, int) and ctx.gpu_id >= 0
    assert torch.cuda.is_available()
    ctx.device = torch.device('cuda', ctx.gpu_id)


@UnifiedInferRunner.register_hook('infer_begin', dependency='prepare_setup')
def prepare_model(ctx):
    ctx.detect_model = R.MODELS.create(ctx.detect_config.model).to(ctx.device)
    ctx.detect_model.eval()
    ctx.dtype = ctx.detect_model.dtype
    ctx.logger.info(f"Built detect model {ctx.detect_config.model.type}")

    if ctx.pose_config:
        ctx.pose_model = R.MODELS.create(ctx.pose_config.model).to(ctx.device)
        ctx.pose_model.eval()
        ctx.logger.info(f"Built pose model {ctx.pose_config.model.type}")


@UnifiedInferRunner.register_hook('infer_begin', dependency='prepare_model')
def prepare_data(ctx):
    data_cfg = ctx.detect_config.data
    if data_cfg.has('data_source'):
        data_cfg.infer.update_at_key('data_source', data_cfg.data_source, overwrite=False)
    if data_cfg.has('data_params'):
        data_cfg.infer.update_at_key('data_params', data_cfg.data_params, overwrite=False)

    detect_dataset_cfg = data_cfg.infer.to_dict()
    detect_dataloader_cfg = detect_dataset_cfg.pop('data_loader', {})
    ctx.detect_dataset = R.DATASETS.create(detect_dataset_cfg)
    ctx.detect_dataloader = D.get_dataloader(ctx.detect_dataset, detect_dataloader_cfg)

    if ctx.pose_config:
        data_cfg = ctx.pose_config.data
        if data_cfg.has('data_source'):
            data_cfg.infer.update_at_key('data_source', data_cfg.data_source, overwrite=False)
        if data_cfg.has('data_params'):
            data_cfg.infer.update_at_key('data_params', data_cfg.data_params, overwrite=False)

        pose_dataset_cfg = data_cfg.infer.to_dict()
        pose_dataloader_cfg = pose_dataset_cfg.pop('data_loader', {})
        ctx.pose_dataset = R.DATASETS.create(pose_dataset_cfg)
        ctx.pose_dataloader = D.get_dataloader(ctx.pose_dataset, pose_dataloader_cfg)


@UnifiedInferRunner.register_hook('infer_begin', dependency='prepare_data')
def prepare_infer(ctx):
    pass


@UnifiedInferRunner.register_hook('infer_iter_steps')
def iterate_infer_step(ctx):
    for i, detect_batch in enumerate(ctx.detect_dataloader):
        yield {'infer_step': i,
               'detect_batch': detect_batch,
               'detect_batch_size': detect_batch['batch_size']}


@UnifiedInferRunner.register_hook('infer_step_begin')
def prepare_infer_step(ctx):
    pass


@UnifiedInferRunner.register_hook('infer_step')
def do_infer_step(ctx):
    with torch.no_grad():
        ctx.detect_batch['inputs'] = U.to(ctx.detect_batch['inputs'], ctx.device, ctx.dtype)
        ctx.detect_results = ctx.detect_model.infer_step(ctx.detect_batch['inputs'], ctx.detect_batch)

        if ctx.has('pose_model'):
            ctx.pose_batch = ctx.pose_dataset.get_input_batch(ctx.detect_batch, ctx.detect_results)
            ctx.pose_batch['inputs'] = U.to(ctx.pose_batch['inputs'], ctx.device, ctx.dtype)
            ctx.pose_results = ctx.pose_model.infer_step(ctx.pose_batch['inputs'], ctx.pose_batch)


@UnifiedInferRunner.register_hook('infer_step_end')
def finish_infer_step(ctx):
    detect_results = ctx.detect_results
    pose_results = ctx.pose_results if ctx.has('pose_results') else None
    results = ctx.detect_dataset.merge_results(detect_results, pose_results)

    display = ctx.options.display if ctx.options.has('display') else False
    sink_id = ctx.options.sink_id if ctx.options.has('sink_id') else 0
    ctx.detect_dataset.visualize(results, ctx.detect_batch, display=display, sink_id=sink_id, work_dir=ctx.work_dir)
    ctx.detect_dataset.evaluate_step(results, ctx.detect_batch, sink_id=sink_id, work_dir=ctx.work_dir, verbose=ctx.verbose)


@UnifiedInferRunner.register_hook('infer_end')
def finish_infer(ctx):
    ctx.detect_dataset.summarize()
    cv2.destroyAllWindows()
