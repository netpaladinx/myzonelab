from .utils import Dict


Context = Dict


class DummyContext:
    pass


class ContextError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(msg)
