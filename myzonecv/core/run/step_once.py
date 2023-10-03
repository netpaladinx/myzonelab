from ..registry import RUNS
from .base_run import BaseRun


@RUNS.register_class('step_once')
class StepOnce(BaseRun):
    def __init__(self, name='step_once', stage_names=None):
        super().__init__(name, stage_names)

    def __call__(self, ctx):
        self.call_stage('run_begin', ctx)
        self.call_stage('step', ctx)
        self.call_stage('run_end', ctx)
