import numpy as np

class Var:
    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value
        self.grad = None
        self.producer = None

    def link_producer(self, func):
        self.producer = func

    def backward(self):
        funcs = [self.producer]
        while funcs:
            func = funcs.pop()
            x, y = func.input_var, func.output_var
            x.grad = func.backward(y.grad)

            if x.producer is not None:
                funcs.append(x.producer)

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
        y.link_producer(self)
        self.output_var = y
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
z.backward()
# y.grad = s2.backward(z.grad)
# x.grad = s1.backward(y.grad)
print(x.grad)
