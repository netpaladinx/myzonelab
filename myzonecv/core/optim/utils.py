import math


def cosine_anneal(start, end, t, weight=1):
    """ t: 0 -> 1
        ret: start -> end, or start * weight + end * (1 - weight) -> end
    """
    t = (math.cos(math.pi * t) + 1) * 0.5 * weight
    return start * t + end * (1 - t)


def linear_anneal(start, end, t):
    """ t: 0 -> 1
        ret: start -> end
    """
    return start * (1 - t) + end * t


def get_momentum(param_group):
    if 'momentum' in param_group:
        return param_group['momentum']

    if 'betas' in param_group:
        return param_group['betas'][0]

    return None


def get_max_epochs_and_max_steps(ctx):
    assert ctx.has('max_epochs')
    if ctx.has('max_steps'):
        return ctx.max_epochs, ctx.max_steps
    else:
        assert ctx.has('steps_per_epoch')
        max_steps = ctx.steps_per_epoch * ctx.max_epochs
        return ctx.max_epochs, max_steps


def get_train_step(ctx):
    if ctx.has('train_step'):
        return ctx.train_step
    else:
        assert ctx.has('step')
        return ctx.step
