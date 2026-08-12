import numpy as np
from christorch.core import Var
from christorch.layers import Module, Linear
from christorch.functions import sigmoid, mean_squared_error
from christorch.optimizers import SGD

# def f(x):
#     s1 = Sin()
#     s2 = Sin()
#     return s2(s1(x))

# def add(x0, x1):
#     return Add()(x0, x1)

# def my_func(x):
#     return Exp()(x)

# # x0 = Var(np.random.randn(2, 3, 5))
# # x1 = Var(np.array([[1.0,2.0],[4.0,5.0], [10.0, 15.0]]))
# # my_mul = lambda x: my_func(x, x1)
# # x1 = Var(np.array([10.0]))
# # my_add = lambda x: add(x0, x)
# # y = x0.reshape(6)
# # y.backward()
# # print(x0.grad)
# # gradient_check(my_func, x0)
# # gradient_check(my_func, x0)

# A = np.array(
#     [
#         [1, 2, 3],
#         [4, 5, 6],
#     ],
#     dtype=np.float32
# )

# B = np.array(
#     [
#         [1, 2],
#         [3, 4],
#         [5, 6],
#     ],
#     dtype=np.float32
# )


# expected = A @ B
# actual = matmul(A, B)

# print("NumPy:")
# print(expected)

# print("AdaptiveCpp:")
# print(actual)

# assert np.allclose(expected, actual)

# print("PASS")

class MyNet(Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = Linear(hidden_size)
        self.l2 = Linear(out_size)

    def forward(self, x):
        z = self.l1(x)
        z = sigmoid(z)
        z = self.l2(z)
        return z

lr = 0.01
iters = 5000

np.random.seed(0)
x = Var(np.random.randn(1000, 1))
y = np.square(x) + np.random.randn(1000, 1)

model = MyNet(10, 1)
optimizer = SGD(model, lr)

for i in range(iters):
    y_pred = model(x)
    loss = mean_squared_error(y, y_pred)
    model.clear_grads()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        print(loss.value)