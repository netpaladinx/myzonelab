from abc import ABCMeta


class BasePlugin(metaclass=ABCMeta):
    def call_before_run(self, ctx):
        pass

    def call_before_epoch(self, ctx):
        pass

    def call_before_step(self, ctx):
        pass

    def call_after_step(self, ctx):
        pass

    def call_after_epoch(self, ctx):
        pass

    def call_after_run(self, ctx):
        pass
