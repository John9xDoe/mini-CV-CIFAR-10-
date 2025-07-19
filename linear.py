import numpy as np
import helper

class Linear:
    def __init__(self, in_features, out_features, mode, lr=0.01):
        self.in_features = in_features
        self.out_features = out_features
        self.lr = lr

        self.W = np.random.randn(out_features, in_features) \
                 * helper.init_weights(mode=mode, fan_in=in_features, fan_out=out_features)

        self.b = np.zeros(out_features)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, d_out):

        self.dW = d_out.T @ self.x # dL/dW = dL/dz * dz/dW
        self.db = np.sum(d_out, axis=0)

        return d_out @ self.W

    def init(self):
        self.W -= self.dW.T * self.lr
        self.b -= self.db.T * self.lr




