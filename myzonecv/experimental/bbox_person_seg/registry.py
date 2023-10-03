from myzonecv.core.registry import Registry, DATA_TRANSFORMS, MODELS, HEADS, LOSSES, POSTPROCESSORS

SEG_TRANSFORMS = Registry('seg_transform', parent=DATA_TRANSFORMS)
SEG_MODELS = Registry('seg_model', parent=MODELS)
SEG_HEADS = Registry('seg_head', parent=HEADS)
SEG_LOSSES = Registry('seg_loss', parent=LOSSES)
SEG_POSTPROCESSORS = Registry('seg_postprocess', parent=POSTPROCESSORS)
