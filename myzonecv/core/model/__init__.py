from . import initializers
from . import bricks
from . import backbones
from . import generators
from . import heads
from . import losses
from . import postprocessors
from . import models


from ..registry import BLOCKS, BACKBONES, GENERATORS, HEADS, LOSSES
from .base_module import BaseModule, Sequential, ModuleList, create_sequential_if_list, create_modulelist_if_list
from .base_model import BaseModel
from .postprocessors import BaseProcess

__all__ = [
    'BaseModule', 'Sequential', 'ModuleList',
    'BaseModel',
    'BaseProcess'
]

BLOCKS.init(create_modulelist_if_list)
BACKBONES.init(create_sequential_if_list)
GENERATORS.init(create_modulelist_if_list)
HEADS.init(create_modulelist_if_list)
LOSSES.init(create_modulelist_if_list)
