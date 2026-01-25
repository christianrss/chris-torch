import numpy as np

class Function:
    def __call__(self, x):
        y = self.forward(x)
        return y

    def forward(self, x):
        raise NotImplementedError()

class Sin(Function):
    def forward(self, x):
        return np.sin(x)

def sin(x):
    return Sin()(x)

x = np.array(np.pi/2)
y = sin(x)
print(y)