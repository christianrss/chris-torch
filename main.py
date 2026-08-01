from tkinter import Variable

import numpy as np

def add_func(funcs, funcs_set, func):
    if func not in funcs_set:
        funcs.append(func)
        funcs_set.add(func)
        funcs.sort(key=lambda x: x.level)

def sum_to(x, out_shape):
    delta_dim = x.ndim - len(out_shape)
    if delta_dim < 0:
        raise ValueError('sum cannot the specific shape, the current input shape is less than the output one')

    delta_axis = tuple(range(delta_dim))
    axis = tuple([i + delta_dim for i, s in enumerate(out_shape) if s == 1])
    y = x.sum(delta_axis + axis, keepdims=True)
    if delta_dim > 0:
        y = y.squeeze(delta_axis)

    if y.shape != out_shape:
        raise ValueError('sum cannot output the specific shape')

    return y

class Var:
    __array_priority__ = 1000000

    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value
        self.grad = None
        self.producer = None
        self.level = 0

    def __add__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Add()(self, other)

    def __radd__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Add()(self, other)

    def __sub__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Sub()(other, self)

    def __rsub__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Sub()(self, other)

    def __mul__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Mul()(self, other)

    def __rmul__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Mul()(self, other)

    def __truediv__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Div()(self, other)

    def __rtruediv__(self, other):
        other = to_array(other)
        other = to_var(other)
        return Div()(other, self)

    def __neg__(self):
        return Neg()(self)

    def __pow__(self, exp):
        return Pow(exp)(self)

    def link_producer(self, func):
        self.producer = func
        self.level = func.level + 1

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.value)

        funcs = []
        funcs_set = set()
        add_func(funcs, funcs_set, self.producer)

        while funcs:
            func = funcs.pop()
            gys = [output_var.grad for output_var in func.output_vars]
            gxs = func.backward(*gys)
            gxs = to_tuple(gxs)

            for x, gx in zip(func.input_vars, gxs):
                x.grad = gx if x.grad is None else x.grad + gx
                if x.producer is not None:
                    add_func(funcs, funcs_set, x.producer)

            for output_var in func.output_vars:
                output_var.gard = None

    def clear_grad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Reshape(shape)(self)

def to_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def to_tuple(x):
    if not isinstance(x, tuple):
        return (x, )
    return x

def to_var(obj):
    if not isinstance(obj, Var):
        return Var(obj)
    return obj

class Function:
    def __call__(self, *xs):
        self.input_vars = xs
        x_values = [x.value for x in xs]
        y_values = self.forward(*x_values)
        y_values = to_tuple(y_values)
        ys = [Var(to_array(y_value)) for y_value in y_values]
        self.level = max([x.level for x in xs])
        for y in ys:
            y.link_producer(self)
        self.output_vars = ys
        return ys if len(ys) > 1 else ys[0]

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
        self.x0_shape, self.x1_shape= x0.shape, x1.shape
        y = x0 + x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape!= self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape
        y = x0 - x1
        return y

    def backward(self, gy):
        gx0, gx1 = gy, -gy
        if self.x0_shape!= self.x1_shape:
            gx0 = sum_to(gx0, self.x0_shape)
            gx1 = sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.input_vars[0].value, self.input_vars[1].value
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape!= x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y

    def backward(self, gy):
        x0, x1 = self.input_vars[0].value, self.input_vars[1].value
        gx0 = gy / x1
        gx1 = gy * (-x0 / x1 ** 2)
        if x0.shape!= x1.shape:
            gx0 = sum_to(gx0, x0.shape)
            gx1 = sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

class Pow(Function):
    def __init__(self, exp):
        self.exp = exp

    def forward(self, x):
        y = x ** self.exp
        return y

    def backward(self, gy):
        x = self.input_vars[0].value
        n = self.exp
        return n * x ** (n - 1) * gy

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape
        self.input_shape = None

    def forward(self, x):
        self.input_shape = x.shape
        y = x.reshape(self.shape)
        return y

    def backward(self, gy):
        return gy.reshape(self.input_shape)

def sin(x):
    return Sin()(x)

def numerical_diff(f, x, h=1e-4):
    x_value = x.value
    grads = np.zeros_like(x_value)
    it = np.nditer(x_value, flags=['multi_index'], op_flags=[['readwrite']])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x_value[idx].copy()
        x_value[idx] = tmp_val - h
        y0 = f(x)
        y0_value = y0.value.copy()

        x_value[idx] = tmp_val + h
        y1 = f(x)
        y1_value = y1.value.copy()
        grads[idx] = (y1_value - y0_value).sum() / (2 * h)
        x_value[idx] = tmp_val
        it.iternext()

    return grads

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

def my_func(x):
    return x.reshape(6)

x0 = Var(np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
x1 = Var(np.array([10.0]))
my_add = lambda x: add(x0, x)
# y = x0.reshape(6)
# y.backward()
# print(x0.grad)
# gradient_check(my_func, x0)
gradient_check(my_add, x1)