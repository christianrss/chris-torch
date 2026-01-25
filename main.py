import numpy as np

class Var:
    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value

class Function:
    def __call__(self, x):
        y = self.forward(x)
        return y

    def forward(self, x):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

def sin(x):
    return Sin()(x)

x = Var(np.array(100))
print(x.value)