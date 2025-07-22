import numpy as np

from linear import Linear


class SGD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, model, lr=None):
        lr = lr if lr is not None else self.lr

        for layer in model.layers:
            if isinstance(layer, Linear):
                layer.W -= lr * layer.dW
                layer.b -= lr * layer.db

class Momentum:
    def __init__(self, model, lr, y=0.9):
        self.y = y
        self.lr = lr

        self.v_W = {}
        self.v_b = {}

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                self.v_W[idx] = np.zeros_like(layer.W)
                self.v_b[idx] = np.zeros_like(layer.b)

    def step(self, model, lr=None):
        lr = lr if lr is not None else self.lr

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                 self.v_W[idx] = self.y * self.v_W[idx] + (1 - self.y) * lr * layer.dW
                 self.v_b[idx] = self.y * self.v_b[idx] + (1 - self.y) * lr * layer.db
                 layer.W -= self.v_W[idx]
                 layer.b -= self.v_b[idx]

class RMSProp:
    def __init__(self, model, lr, p=0.9, eps=1e-8):
        self.p = p
        self.eps = eps
        self.lr = lr

        self.G_W = {}
        self.G_b = {}

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                self.G_W[idx] = np.zeros_like(layer.W)
                self.G_b[idx] = np.zeros_like(layer.b)

    def step(self, model, lr=None):
        lr = lr if lr is not None else self.lr

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                self.G_W[idx] = self.p * self.G_W[idx] + (1 - self.p) * layer.dW ** 2
                self.G_b[idx] = self.p * self.G_b[idx] + (1 - self.p) * layer.db ** 2
                layer.W -= lr * layer.dW / (np.sqrt(self.G_W[idx]) + self.eps)
                layer.b -= lr * layer.db / (np.sqrt(self.G_b[idx]) + self.eps)

class Adam:
    def __init__(self, model, lr, y=0.9, a=0.999, eps=10**(-8)):
        self.lr = lr
        self.y = y
        self.a = a
        self.eps = eps
        self.t = 0

        self.v_W, self.v_b, self.G_W, self.G_b = {}, {}, {}, {}

        for idx, layer in enumerate(model.layers):
            if isinstance(layer, Linear):
                self.v_W[idx] = np.zeros_like(layer.W)
                self.v_b[idx] = np.zeros_like(layer.b)
                self.G_W[idx] = np.zeros_like(layer.W)
                self.G_b[idx] = np.zeros_like(layer.b)

    def step(self, model, lr=None):
        lr = lr if lr is not None else self.lr
        self.t += 1

        for idx, layer in enumerate(model.layers):
            # momentum
            self.v_W[idx] = self.y * self.v_W[idx] + (1 - self.y) * layer.dW
            self.v_b[idx] = self.y * self.v_b[idx] + (1 - self.y) * layer.db

            # RMS
            self.G_W[idx] = self.a * self.G_W[idx] + (1 - self.a) * layer.dW ** 2
            self.G_b[idx] = self.a * self.G_b[idx] + (1 - self.a) * layer.db ** 2

            # Bias correction
            v_W_corr = self.v_W[idx] / (1 - self.y ** self.t)
            v_b_corr = self.v_b[idx] / (1 - self.y ** self.t)
            G_W_corr = self.G_W[idx] / (1 - self.a ** self.t)
            G_b_corr = self.G_b[idx] / (1 - self.a ** self.t)

            layer.W -= lr * (v_W_corr / (np.sqrt(G_W_corr) + self.eps))
            layer.b -= lr * (v_b_corr / (np.sqrt(G_b_corr) + self.eps))