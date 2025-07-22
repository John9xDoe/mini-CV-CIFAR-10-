import json
import logging

import numpy as np

from linear import  Linear
from activations import ReLu, Softmax
from model_context import ExperimentContext


class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, activation_class):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation_class = activation_class

        self.layers = [
            Linear(input_dim, self.hidden_dim, mode='He'),
            self.activation_class(),
            Linear(self.hidden_dim, output_dim, mode='Xe'),
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
            "ReLu": ReLu,
            # "Sigmoid": Sigmoid
        }[name]

    def save_model(self):
        params = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                params[f"W{i}"] = layer.W
            if hasattr(layer, 'b'):
                params[f"b{i}"] = layer.b

        meta={
            "cnt_layers": sum(isinstance(l, Linear) for l in self.layers),
            "hidden_size": self.hidden_dim,
            "activation": self.activation_class.__name__,
        }

        ctx = ExperimentContext()
        np.savez(ctx.get_path("model.npz"), **params)
        with open(ctx.get_path("config.json"), 'w') as f:
            json.dump(meta, f)

        logging.info(
            f"""model saved:
            layers_cnt: {meta["cnt_layers"]}
            hidden_size: {meta["hidden_size"]}
            activation: {meta["activation"]}
            """
        )

    @classmethod
    def load_model(cls, folder_id, input_dim=3072, output_dim=10):
        data = np.load(f"experiments/model_{folder_id}/model.npz")
        with open(f"experiments/model_{folder_id}/config.json", 'r') as f:
            meta = json.load(f)

        activation_class = MLP.get_activation_class_by_name(meta["activation"])
        model = cls(input_dim, meta["hidden_size"], output_dim, activation_class)

        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'W'):
                layer.W = data[f"W{i}"]
            if hasattr(layer, 'b'):
                layer.b = data[f"b{i}"]

        logging.info(
            f"""model loaded: 
            input_size: {model.input_dim} (default)
            hidden_size: {model.hidden_dim}
            output_size: {model.output_dim} (default)
            activation_class: {model.activation_class.__name__} 
            layers_count: {meta['cnt_layers']}
            """
        )
        return model



