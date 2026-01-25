import numpy as np

class Var:
    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value
        self.grad = None

def to_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, x):
        self.input_var = x
        x_value = x.value
        y_value = self.forward(x_value)
        y_value = to_array(y_value)
        y = Var(y_value)
        return y

    def forward(self, x_value):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x_value):
        return np.sin(x_value)

    def backward(self, gy):
        x_value = self.input_var.value
        gx = np.cos(x_value) * gy
        return gx

def sin(x):
    return Sin()(x)

# x = np.array(2)
# y = to_array(np.square(x))
# print(y, type(y))

x = Var(np.array(np.pi/2))
s1 = Sin()
s2 = Sin()
y = s1(x)
z = s2(y)

z.grad = np.array(1.0)
y.grad = s2.backward(z.grad)
s1.backward(y.grad)
x.grad = s1.backward(y.grad)
print(x.grad)
