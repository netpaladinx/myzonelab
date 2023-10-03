from ..registry import RUNNERS
from ..runner import Runner


@RUNNERS.register_class('train')
class TrainRunner(Runner):
    run_cfg = {
        'type': 'epoch_step_loop',
        'name': 'train',
        'stage_names': {
            'run_begin': 'train_begin',
            'iter_epochs': 'train_iter_epochs',
            'epoch_begin': 'train_epoch_begin',
            'iter_steps': 'train_iter_steps',
            'step_begin': 'train_step_begin',
            'step': 'train_step',
            'step_end': 'train_step_end',
            'epoch_end': 'train_epoch_end',
            'run_end': 'train_end'
        }
    }


@RUNNERS.register_class('val')
class ValRunner(Runner):
    run_cfg = {
        'type': 'step_loop',
        'name': 'val',
        'stage_names': {
            'run_begin': 'val_begin',
            'iter_steps': 'val_iter_steps',
            'step_begin': 'val_step_begin',
            'step': 'val_step',
            'step_end': 'val_step_end',
            'run_end': 'val_end'
        }
    }


@RUNNERS.register_class('test')
class TestRunner(Runner):
    run_cfg = {
        'type': 'step_loop',
        'name': 'test',
        'stage_names': {
            'run_begin': 'test_begin',
            'iter_steps': 'test_iter_steps',
            'step_begin': 'test_step_begin',
            'step': 'test_step',
            'step_end': 'test_step_end',
            'run_end': 'test_end'
        }
    }


@RUNNERS.register_class('infer')
class InferRunner(Runner):
    run_cfg = {
        'type': 'step_loop',
        'name': 'infer',
        'stage_names': {
            'run_begin': 'infer_begin',
            'iter_steps': 'infer_iter_steps',
            'step_begin': 'infer_step_begin',
            'step': 'infer_step',
            'step_end': 'infer_step_end',
            'run_end': 'infer_end'
        }
    }
    
    
@RUNNERS.register_class('unified_infer')
class UnifiedInferRunner(Runner):
    run_cfg = {
        'type': 'step_loop',
        'name': 'unified_infer',
        'stage_names': {
            'run_begin': 'infer_begin',
            'iter_steps': 'infer_iter_steps',
            'step_begin': 'infer_step_begin',
            'step': 'infer_step',
            'step_end': 'infer_step_end',
            'run_end': 'infer_end'
        }
    }


@RUNNERS.register_class('image_infer')
class ImageInferRunner(Runner):
    run_cfg = {
        'type': 'step_once',
        'name': 'image_infer',
        'stage_names': {
            'run_begin': 'infer_begin',
            'step': 'infer_step',
            'run_end': 'infer_end'
        }
    }


@RUNNERS.register_class('torch2onnx')
class Torch2ONNXRunner(Runner):
    run_cfg = {
        'type': 'step_once',
        'name': 'torch2onnx',
        'stage_names': {
            'run_begin': 'export_begin',
            'step': 'export_step',
            'run_end': 'export_end'
        }
    }


@RUNNERS.register_class('complex_train')
class ComplexTrainRunner(Runner):
    run_cfg = {
        'type': 'station_thread_loop',
        'name': 'complex_train',
        'stage_names': {
            'run_begin': 'train_begin',
            'enter_station': 'enter_station',
            'enter_thread': 'enter_thread',
            'execute_thread': 'execute_thread',
            'exit_thread': 'exit_thread',
            'exit_station': 'exit_station',
            'run_end': 'train_end'
        }
    }
    run_scheduler_cfg = {
        'type': 'station_thread_loop_scheduler',
    }
