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

def sum_backward_shape(gy, x_shape, axis, keepdims):
    x_dim = len(x_shape)
    if x_dim > 0 and axis is not None and not keepdims:
        axis = to_tuple(axis)
        axis = [a if a >= 0 else a + x_dim for a in axis]
        shape = list(gy.shape)
        for a in sorted(axis):
            shape.insert(a, 1)
    else:
        shape = gy.shape

    return shape

def to_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def to_tuple(x):
    if not isinstance(x, tuple):
        return (x, )
    return x