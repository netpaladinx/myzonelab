from myzonecv.core.registry import Registry, DATA_TRANSFORMS, MODELS, HEADS, LOSSES, POSTPROCESSORS

COLORADJUST_TRANSFORMS = Registry('coloradjust_transform', parent=DATA_TRANSFORMS)
COLORADJUST_MODELS = Registry('coloradjust_model', parent=MODELS)
COLORADJUST_HEADS = Registry('coloradjust_head', parent=HEADS)
COLORADJUST_LOSSES = Registry('coloradjust_loss', parent=LOSSES)
COLORADJUST_POSTPROCESSORS = Registry('coloradjust_postprocess', parent=POSTPROCESSORS)
