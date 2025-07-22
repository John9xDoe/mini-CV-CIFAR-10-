import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model):
        for layer in model.layers:
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

class Momentum:
    def __init__(self, model, lr=0.01, y=0.9):
        self.lr = lr
        self.y = y

        self.v_W = {}
        self.v_b = {}

        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                self.v_W[idx] = np.zeros_like(layer.W)
                self.v_b[idx] = np.zeros_like(layer.b)

    def step(self, model):
        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                 self.v_W[idx] = self.y * self.v_W + (1 - self.y) * self.lr * layer.dW
                 self.v_b[idx] = self.y * self.v_b + (1 - self.y) * self.lr * layer.db
                 layer.W -= self.v_W[idx]
                 layer.b -= self.v_b[idx]

class RMSProp:
    def __init__(self, model, lr=0.01, p=0.9, eps=1e-8):
        self.lr = lr
        self.p = p
        self.eps = eps

        self.G_W = {}
        self.G_b = {}

        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                self.G_W[idx] = np.zeros_like(layer.dW)
                self.G_b[idx] = np.zeros_like(layer.db)

    def step(self, model):
        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                self.G_W[idx] = self.p * self.G_W[idx] + (1 - self.p) * layer.dW ** 2
                self.G_b[idx] = self.p * self.G_b[idx] + (1 - self.p) * layer.db ** 2
                layer.W -= self.lr * layer.dW / (np.sqrt(self.G_W[idx]) + self.eps)
                layer.b -= self.lr * layer.db / (np.sqrt(self.G_b[idx]) + self.eps)

class Adam:
    def __init__(self, model, lr=0.01, y=0.9, a=0.999, eps=10**(-8)):
        self.lr = lr
        self.y = y
        self.a = a
        self.eps = eps
        self.t = 0

        self.v_W, self.v_b, self.G_W, self.G_b = {}, {}, {}, {}

        for idx, layer in enumerate(model.layers):
            if hasattr(layer, 'W') and hasattr(layer, 'dW'):
                self.v_W[idx] = np.zeros_like(layer.W)
                self.v_b[idx] = np.zeros_like(layer.b)
                self.G_W[idx] = np.zeros_like(layer.W)
                self.G_b[idx] = np.zeros_like(layer.b)

    def step(self, model):
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

            layer.W -= self.lr * (v_W_corr / (np.sqrt(G_W_corr) + self.eps))
            layer.b -= self.lr * (v_b_corr / (np.sqrt(G_b_corr) + self.eps))