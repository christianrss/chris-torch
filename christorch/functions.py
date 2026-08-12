import numpy as np
from christorch.core import Sin, MatMul, to_var, Exp, Sum

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

def linear(x, w, b=None):
    temp = MatMul()(x, w)
    if b is None:
        return temp

    y = temp + b
    return y

def sigmoid(x):
    x = to_var(x)
    y = 1 / (1 + Exp()(-x))
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    return Sum()(diff ** 2) / len(diff)
