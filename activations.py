import numpy as np

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