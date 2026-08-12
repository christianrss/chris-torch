import numpy as np
from christorch.helper import to_array, to_tuple, add_func, sum_to, sum_backward_shape
from christorch.backend.acpp import matmul

class Var:
    __array_priority__ = 1000000

    def __init__(self, value):
        if value is not None and not isinstance(value, np.ndarray):
            raise TypeError("The value must be a ndarray")
        self.value = value
        self.grad = None
        self.producer = None
        self.level = 0

    @property
    def shape(self):
        return self.value.shape

    def __len__(self):
        return len(self.value)

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
                output_var.grad = None

    def clear_grad(self):
        self.grad = None

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return Reshape(shape)(self)

    def transpose(self, *axes):
        if len(axes) == 0:
            axes = None
        elif len(axes) == 1:
            if isinstance(axes[0], (tuple, list)) or axes[0] is None:
                axes = axes[0]
        return Transpose(axes)(self)


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

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        y = self.output_vars[0].value
        return y * gy

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

class Sum(Function):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y

    def backward(self, gy):
        gy_shape = sum_backward_shape(gy, self.x_shape, self.axis, self.keepdims)
        gy = gy.reshape(gy_shape)
        gx = np.broadcast_to(gy, self.x_shape)
        return gx

# Acpp matmul
class MatMul(Function):
    def forward(self, x, W):
        return matmul(x, W)

    def backward(self, gy):
        x = self.input_vars[0].value
        W = self.input_vars[1].value

        gx = matmul(
            gy,
            np.ascontiguousarray(W.T)
        )

        gW = matmul(
            np.ascontiguousarray(x.T),
            gy
        )

        return gx, gW
# class MatMul(Function):
#     def forward(self, x, W):
#         y = np.dot(x, W)
#         return y

#     def backward(self, gy):
#         x, W = self.input_vars[0].value, self.input_vars[1].value
#         gx = np.dot(gy, W.T)
#         gW = np.dot(x.T, gy)
#         return gx, gW

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        y = x.transpose(self.axes)
        return y

    def backward(self, gy):
        if self.axes is None:
            return gy.transpose()

        axes_index = np.argsort([axis if axis >= 0 else axis + len(self.axes) for axis in self.axes])
        return gy.transpose(axes_index)
