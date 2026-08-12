class Optimizer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr

    def step(self):
        for param in self.model.params():
            if param.grad is not None:
                self.update_param(param)

    def update_param(self, param):
        raise NotImplementedError()

class SGD(Optimizer):
    def update_param(self, param):
        param.value -= self.lr * param.grad
