import time

import numpy as np
import logging
import yaml

import helper
import metrics
from modules.losses import CrossEntropy
from modules.metrics import accuracy
from modules.model_context import ExperimentContext
from optimizers import Momentum, SGD, RMSProp, Adam
from visualisations import Visualizer


class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size, lr):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr

    def train(self, epochs, X_train, y_train, X_test=None, y_test=None, log_interval=1, timer=False, graph=False):

        self.epochs = epochs
        if timer:
            start_time = time.time()

        accs, epchs = [], []

        for epoch in range(epochs):
            for x_batch, y_batch in zip(*helper.get_batches(X_train, y_train, self.batch_size)):
                logits = self.model.forward(x_batch)

                loss = self.loss_fn.forward(logits, y_batch)
                d_out = self.loss_fn.backward()

                self.model.backward(d_out)
                self.optimizer.step(self.model, self.lr)

            log_msg = f"epoch {epoch}/{epochs - 1} has passed"

            if epoch % log_interval == 0:
                y_pred = self.model.predict(X_test)
                log_msg += f": loss={loss:.4f}, accuracy={metrics.accuracy(y_pred, y_test):.2f}%"

            if graph:
                epchs.append(epoch + 1)
                accs.append(metrics.accuracy(y_pred, y_test))

            logging.info(log_msg)

        if timer:
            logging.info(f"training time: {time.time() - start_time} s.")

        if graph:
            Visualizer.plot_epochs_currency(epchs, accs)


    def save_config(self):
        args = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.epochs,
            "loss_fn": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__
        }

        ctx = ExperimentContext()

        with open (ctx.get_path('args.yaml'), 'w') as f:
            yaml.dump(args, f)

        logging.info(
            f"""config saved:
            batch_size: {args["batch_size"]}
            learning_rate: {args["lr"]}
            epochs: {args["epochs"]}
            loss_fn: {args["loss_fn"]}
            optimizer: {args["optimizer"]}
            """
        )

    @staticmethod
    def load_config(folder_id):
        loss_fns = {
            "CrossEntropy": CrossEntropy()
        }

        optimizers = {
            "SGD": SGD(),
            "Momentum": Momentum(),
            "RMSProp": RMSProp(),
            "Adam": Adam()
        }
        with open (f"experiments/model_{folder_id}/args.yaml", 'r') as f:
            args = yaml.safe_load(f)

        args["loss_fn"] = loss_fns[args["loss_fn"]]
        args["optimizer"] = optimizers[args["optimizer"]]

        logging.info(
            f"""config loaded:
            batch_size: {args["batch_size"]}
            learning_rate: {args["lr"]}
            epochs: {args["epochs"]}
            loss_fn: {args["loss_fn"]}
            optimizer: {args["optimizer"]}
            """
        )

        return args