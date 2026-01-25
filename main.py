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
        if self.grad is None:
            self.grad = np.array(1.0)
        funcs = [self.producer]
        while funcs:
            func = funcs.pop()
            gys = [output_var.grad for output_var in func.output_vars]
            gxs = func.backward(*gys)
            gxs = to_tuple(gxs)

            for x, gx in zip(func.input_vars, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
                if x.producer is not None:
                    funcs.append(x.producer)
    def clear_grad(self):
        self.grad = None

def to_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def to_tuple(x):
    if not isinstance(x, tuple):
        return (x, )
    return x

class Function:
    def __call__(self, *xs):
        self.input_vars = xs
        x_values = [x.value for x in xs]
        y_values = self.forward(*x_values)
        y_values = to_tuple(y_values)
        ys = [Var(to_array(y_value)) for y_value in y_values]
        for y in ys:
            y.link_producer(self)
        self.output_vars = ys
        return ys if  len(ys) > 1 else ys[0]

    def forward(self, *x_values):
        raise NotImplementedError()

    def backward(self, *gys):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x_value):
        return np.sin(x_value)

    def backward(self, gy):
        x_value = self.input_vars[0].value
        gx = np.cos(x_value) * gy
        return gx

class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy

def numerical_diff(f, x, h=1e-4):
    x0 = Var(np.array(x.value - h))
    x1 = Var(np.array(x.value + h))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.value - y0.value) / (2 * h)

def gradient_check(f, x):
    y = f(x)
    y.backward()
    numerical_grad = numerical_diff(f, x)
    if not np.allclose(x.grad, numerical_grad):
        print('gradient check failed')

def f(x):
    s1 = Sin()
    s2 = Sin()
    return s2(s1(x))

def add(x0, x1):
    return Add()(x0, x1)

x0 = Var(np.array(2))
x1 = Var(np.array(3))
my_add = lambda x: add(x, x)
gradient_check(my_add, x0)

x0.clear_grad()
x1.clear_grad()
my_add1 = lambda x: add(x, x1)
gradient_check(my_add1, x0)

# a = Var(np.array(1.0))
# b = Var(np.array(2.0))
# A = Add()
# c = A(a, b)
# print(c.value)