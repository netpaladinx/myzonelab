import time

import thop
import torch
import torch.nn as nn


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def profile(input, module, device=None, n=10, flops_only=False):
    """ Usage:
            input = torch.randn(16, 3, 640, 640)
            m1 = lambda x: x * torch.sigmoid(x)
            m2 = nn.SiLU()
            profile(input, [m1, m2], device, n=100)  # profile over 100 iterations 
    """
    def clean(mod):
        if 'total_ops' in mod._buffers:
            mod._buffers.pop('total_ops')
        if 'total_params' in mod._buffers:
            mod._buffers.pop('total_params')

    results = []
    print(f"{'Params':>12s}{'GFLOPs':>12s}{'GPU_mem (GB)':>14s}{'forward (ms)':>14s}{'backward (ms)':>14s}{'input':>24s}{'output':>24s}")

    for x in (input if isinstance(input, list) else [input]):
        if device is None:
            dev = x.device
        else:
            dev = device
            x = x.to(dev)
        x.requires_grad = True
        input_is_half = x.dtype is torch.float16

        for m in (module if isinstance(module, list) else [module]):
            m = m.to(dev) if hasattr(m, 'to') else m
            m = m.half() if hasattr(m, 'half') and input_is_half else m
            dt_forward, dt_backward, t = 0., 0., [0., 0., 0.]
            prev_training_status = m.training
            try:
                if hasattr(m, 'is_dummy'):
                    m.is_dummy = True

                flops = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2  # GFLOPs

                if hasattr(m, 'is_dummy'):
                    m.is_dummy = False
            except Exception as e:
                flops = 0
                print(e)

            if flops_only:
                results.append(flops)
            else:
                try:
                    for _ in range(n):
                        t[0] = time_sync()
                        y = m(x, is_dummy=True)
                        t[1] = time_sync()

                        try:
                            _ = (sum([yi.sum() for yi in y]) if isinstance(y, list) else y).sum().backward()
                            t[2] = time_sync()
                        except Exception as e:  # no backward method
                            t[2] = float('nan')

                        dt_forward += (t[1] - t[0]) * 1000 / n  # ms per op forward
                        dt_backward += (t[2] - t[1]) * 1000 / n  # ms per op backward

                    mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)

                    s_in = (tuple(x.shape) if isinstance(x, torch.Tensor) else 'list') if n > 0 else None
                    s_out = (tuple(y.shape) if isinstance(y, torch.Tensor) else 'list') if n > 0 else None

                    n_params = sum(list(x.numel() for x in m.parameters())) if isinstance(m, nn.Module) else 0  # parameters

                    print(f'{n_params:12}{flops:12.4g}{mem:>14.3f}{dt_forward:14.4g}{dt_backward:14.4g}{str(s_in):>24s}{str(s_out):>24s}')
                    results.append([n_params, flops, mem, dt_forward, dt_backward, s_in, s_out])

                except Exception as e:
                    print(e)
                    results.append(None)

                torch.cuda.empty_cache()

            m.train(prev_training_status)
            m.apply(clean)
    return results
