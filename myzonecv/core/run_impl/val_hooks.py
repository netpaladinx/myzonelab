import torch

from .. import utils as U
from .runners import ValRunner


@ValRunner.register_hook('val_begin')
def prepare_val(ctx):
    """ ctx from train hooks
    """
    ctx.model.eval()
    ctx.mode = 'val'
    ctx.model.call_before_eval(ctx)

    ctx.val_total_steps = len(ctx.val_dataloader)
    ctx.val_all_results = []


@ValRunner.register_hook('val_iter_steps')
def iterate_val_step(ctx):
    ctx.val_progress_bar = U.ProgressBar(len(ctx.val_dataset), start=True)

    for i, val_batch in enumerate(ctx.val_dataloader):
        yield {'val_step': i,
               'val_batch': val_batch,
               'val_batch_size': val_batch['batch_size']}


@ValRunner.register_hook('val_step_begin')
def prepare_val_step(ctx):
    if ctx.has('diagnoser'):
        ctx.diagnoser.call_before_step(ctx)


@ValRunner.register_hook('val_step')
def do_val_step(ctx):
    with torch.no_grad():
        val_batch = ctx.val_batch
        val_batch['inputs'] = U.to(val_batch['inputs'], ctx.device, ctx.dtype)
        if 'targets' in val_batch:
            val_batch['targets'] = U.to(val_batch['targets'], ctx.device, ctx.dtype)

        results = ctx.model.eval_step(val_batch['inputs'], val_batch)
        ctx.val_results = results
        ctx.val_all_results.append(results)

        ctx.val_progress_bar.update(n=ctx.val_batch_size)


@ValRunner.register_hook('val_step_end')
def finish_val_step(ctx):
    if ctx.has('diagnoser'):
        ctx.diagnoser.call_after_step(ctx)


@ValRunner.register_hook('val_end')
def finish_val(ctx):
    ctx.val_progress_bar.end()

    plot_dir = ctx.plot_dir if ctx.has('plot_dir') else None
    val_results = ctx.val_dataset.evaluate_all(ctx.val_all_results, plot_dir=plot_dir)
    ctx.summarizer.call_to_log(ctx, val_results, ctx.epoch + 1 if ctx.has('epoch') else 0)

    ctx.model.call_after_eval(ctx)
    ctx.model.train()
    ctx.mode = 'train'
