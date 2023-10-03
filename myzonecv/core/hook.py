
class Hook:
    def __init__(self, name, func, dependency=None, stage_name=None, run_name=None):
        self.name = name
        self.func = func
        self.stage_name = stage_name
        self.run_name = run_name
        
        if dependency and not isinstance(dependency, (list, tuple)):
            dependency = [dependency]
        self.dependency = dependency
        
    def __call__(self, ctx):
        return self.func(ctx)
        
    def ready(self, finished):
        if not self.dependency:
            return True
        
        inter = set(self.dependency) & set(finished)
        return len(self.dependency) == len(inter)