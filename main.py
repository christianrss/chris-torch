import numpy as np
import math
from christorch.core import Var
from christorch.layers import Module, Linear
from christorch.functions import sigmoid, mean_squared_error
from christorch.optimizers import SGD

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
x = np.random.randn(30000, 1)
y = np.square(x) + np.random.randn(30000, 1)
data_size = len(x)
batch_size = 100
epochs = 10
iters = math.ceil(data_size / batch_size)

model = MyNet(10, 1)
optimizer = SGD(model, lr)

for epoch in range(epochs):
    total_loss = 0
    index = np.random.permutation(data_size)
    for i in range(iters):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = Var(x[batch_index])
        batch_label = y[batch_index]
        y_pred = model(batch_x)
        loss = mean_squared_error(batch_label, y_pred)
        model.clear_grads()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.value) * len(batch_label)

    avg_loss = total_loss / data_size
    print('epoch {}, loss {}'.format(epoch + 1, avg_loss))