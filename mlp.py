import json
import logging

import numpy as np

from linear import  Linear
from activations import ReLu, Softmax
from model_context import ExperimentContext


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_class, lr):

        self.hidden_dim = hidden_dim
        self.activation_class = activation_class
        self.lr = lr

        self.layers = [
            Linear(input_dim, self.hidden_dim, mode='He', lr=lr),
            self.activation_class(),
            Linear(self.hidden_dim, output_dim, mode='Xe', lr=lr),
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

    @staticmethod
    def get_activation_class_by_name(name):
        return {
            "ReLU": ReLu,
            # "Sigmoid": Sigmoid
        }[name]

    def save_model(self, epochs):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[f"W{i}"] = layer.W
            if hasattr(layer, 'b'):
                params[f"b{i}"] = layer.b

        meta={
            "cnt_layers": sum(isinstance(l, Linear) for l in self.layers),
            "learning_rate": self.lr,
            "hidden_size": self.hidden_dim,
            "activation": self.activation_class.__name__,
            "epochs": epochs
        }

        ctx = ExperimentContext()
        np.savez(ctx.get_path("model.npz"), **params)
        with open(ctx.get_path("config.json"), 'w') as f:
            json.dump(meta, f)

    @classmethod
    def load_model(cls, filename, input_dim=3072, output_dim=10):
        ctx = ExperimentContext()

        data = np.load(ctx.get_path("model.npz"))
        with open(ctx.get_path("config.json"), 'r') as f:
            meta = json.load(f)

        activation_class = MLP.get_activation_class(meta["activation"])
        model = cls(input_dim, meta["hidden_size"], output_dim, activation_class)

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'W'):
                layer.W = data[f"W{i}"]
            if hasattr(layer, 'b'):
                layer.b = data[f"b{i}"]

        logging.info(f"model loaded: {model.input_dim}")
        return model



