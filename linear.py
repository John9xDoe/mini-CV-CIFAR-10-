from functools import lru_cache

import numpy as np
from .main import init_weights

class Linear:
    def __init__(self, in_features, out_features, mode, lr=0.01):
        self.in_features = in_features
        self.out_features = out_features

        self.W = np.random.randn(in_features, out_features) \
                 * init_weights(mode=mode, fan_in=in_features, fan_out=out_features)

        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b

    def backward(self, d_out):

        self.dW = d_out @ self.x # dL/dW = dL/dz * dz/dW
        self.db = np.sum(d_out, axis=0)

        return d_out @ self.x

    def init(self):
        self.W -= self.dW * self.lr
        self.b -= self.db * self.lr

class ReLu:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, d_out):
        return d_out * self.mask

class Softmax:
    def forward(self, logits):
        logits -= np.max(logits, axis=1, keepdims=True)  # explosion protection
        exp = np.exp(logits)
        return exp / np.sum(exp, axis=1, keepdims=True) # axis=1 for bathes, keepdims=1 - shape saving
    def backward(self, d_out): # stub: dL/dz = softmax(logits) - y_true
        # so Softmax.backward() take into account in CrossEntropy.backward()
        return d_out