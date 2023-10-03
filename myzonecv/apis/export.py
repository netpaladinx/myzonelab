from myzonecv.core import Context
from myzonecv.core.registry import RUNNERS


def torch2onnx(config,
               runner='torch2onnx',
               checkpoint=None,
               input_size=None,
               output_file=None,
               device='cpu',
               opset_version=11,
               apply_prerun=False,
               show=False,
               verifty=0,
               **kwargs):
    assert checkpoint is not None
    assert input_size is not None
    assert output_file is not None

    kwargs.update({
        'checkpoint': checkpoint,
        'input_size': input_size,
        'output_file': output_file,
        'device': device,
        'opset_version': opset_version,
        'apply_prerun': apply_prerun,
        'show': show,
        'verifty': verifty
    })

    ctx = Context(config=config,
                  options=kwargs)

    runner = RUNNERS.create({'type': runner})
    runner.run(ctx)
