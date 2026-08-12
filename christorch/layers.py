import numpy as np
from christorch.core import Var
from christorch.helper import to_tuple
from christorch.functions import linear

class Param(Var):
    pass

class Module:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Param, Module)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *xs):
        self.input_vars = xs
        ys = self.forward(*xs)
        ys = to_tuple(ys)
        self.output_vars = ys
        return ys if len(ys) > 1 else ys[0]

    def forward(self, *xs):
        raise NotImplementedError()

    def params(self):
        for name in self._params:
            p = self.__dict__[name]
            if isinstance(p, Module):
                yield from p.params()
            else:
                yield p

    def clear_grads(self):
        for param in self.params():
            param.clear_grad()

class Linear(Module):
    def __init__(self, out_size, in_size=None, bias=True):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.Weight = Param(None)

        if in_size is not None:
            self.init_W()

        if bias:
            self.Bias = Param(np.zeros(out_size))
        else:
            self.Bias = None

    def _init_W(self):
        self.Weight.value = np.random.randn(self.in_size, self.out_size) * np.sqrt(1 / self.in_size)

    def forward(self, x):
        if self.Weight.value is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = linear(x, self.Weight, self.Bias)
        return y