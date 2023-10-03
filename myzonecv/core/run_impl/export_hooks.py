import torch
import onnxruntime as onnxrt
import numpy as np

from ..config import Config
from ..registry import MODELS
from ..utils import load_model, replace_syncbn, remove_initializer_inputs, check_noninitializer_inputs, partial, print_onnx_model
from .runners import Torch2ONNXRunner


@Torch2ONNXRunner.register_hook('export_begin')
def prepare_torch2onnx(ctx):
    # load config
    assert ctx.has('config') and isinstance(ctx.config, (str, Config))
    if isinstance(ctx.config, str):
        ctx.config = Config.from_file(ctx.config)
    config = ctx.config

    # merge options
    assert ctx.has('options')
    config.merge(ctx.options)

    # create model and load checkpoint
    model = MODELS.create(config.model)
    assert config.has('checkpoint') and config.has('device')
    load_model(config.checkpoint, map_location=config.device, model=model)

    # modify model
    model = replace_syncbn(model)
    kwargs_input = config.get('kwargs_input', {})
    model.forward = partial(model.forward, is_dummy=True, **kwargs_input)

    model.to(config.device)
    model.eval()

    ctx.model = model

    print(f"Create a PyTorch model and load the checkpoint from {config.checkpoint}")


@Torch2ONNXRunner.register_hook('export_step')
def do_torch2onnx(ctx):
    config = ctx.config

    input_tensor = torch.randn(config.input_size)
    input_names = config.get('input_names', ['input'])
    output_names = config.get('output_names', ['output'])

    if config.apply_prerun:
        ctx.model.forward(input_tensor)

    torch.onnx.export(ctx.model,
                      input_tensor,
                      config.output_file,
                      export_params=True,
                      keep_initializers_as_inputs=True,
                      verbose=config.show,
                      opset_version=config.opset_version,
                      input_names=input_names,
                      output_names=output_names)

    # only keep feeding input
    remove_initializer_inputs(config.output_file)

    # print_onnx_model(config.output_file)

    print(f"Generate an ONNX model and save it as {config.output_file}")


@Torch2ONNXRunner.register_hook('export_end')
def finish_torch2onnx(ctx):
    def verify(config, torch_model):
        num_verify = config.get('verify', 0)
        if num_verify:
            print("Start to verify ouputs between PyTorch and ONNX Models")
            feed_input = check_noninitializer_inputs(config.output_file)
            sess = onnxrt.InferenceSession(config.output_file)

            for _ in range(num_verify):
                input_tensor = torch.randn(config.input_size)

                # output from pytorch model
                torch_output = torch_model(input_tensor)

                # output from onnx model
                onnx_output = sess.run(None, {feed_input: input_tensor.detach().numpy()})

                if not isinstance(torch_output, (list, tuple)):
                    torch_output = [torch_output]

                assert len(torch_output) == len(onnx_output)
                for torch_out, onnx_out in zip(torch_output, onnx_output):
                    torch_out = torch_out.detach().numpy()
                    assert np.allclose(torch_out, onnx_out, atol=1.e-5)

            print(f"The numerical values are same between PyTorch and ONNX models after verifying {num_verify} times")

    max_tries = 3
    num_tries = 0
    while num_tries < max_tries:
        try:
            verify(ctx.config, ctx.model)
            break
        except AssertionError as err:
            num_tries += 1
            print(f'[TRY {num_tries}] {err}')
            if num_tries == max_tries:
                raise err
