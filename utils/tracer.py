from torch.fx import Tracer

class CustomTracer(Tracer):
    def is_leaf_module(self, m, qualname):
        if type(m).__name__ in ['Conv2d', 'BatchNorm2d', 'Linear']:
            return True
        return super().is_leaf_module(m, qualname)
