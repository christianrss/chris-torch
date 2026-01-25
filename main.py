import numpy as np

class Var:
    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value

def to_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, x):
        x_value = x.value
        y_value = self.forward(x_value)
        y_value = to_array(y_value)
        y = Var(y_value)
        return y

    def forward(self, x_value):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x_value):
        return np.sin(x_value)

def sin(x):
    return Sin()(x)

# x = np.array(2)
# y = to_array(np.square(x))
# print(y, type(y))

x = Var(np.array(np.pi/2))
y = sin(x)
print(y.value)