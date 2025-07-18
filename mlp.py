import numpy as np
from linear import  Linear
from activations import ReLu, Softmax

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.layers = [
            Linear(input_dim, hidden_dim, mode='He'),
            ReLu(),
            Linear(hidden_dim, output_dim, mode='Xe'),
            Softmax()
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)