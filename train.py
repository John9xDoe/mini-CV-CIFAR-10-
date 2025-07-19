import time

import numpy as np
import logging

import helper
import metrics
from linear import Linear
from metrics import accuracy
from visualisations import Visualizer


class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train(self, epochs, X_train, y_train, X_test=None, y_test=None, log_interval=1, timer=False, graph=False):
        if timer:
            start_time = time.time()

        accs, epchs = [], []

        for epoch in range(epochs):
            for x_batch, y_batch in zip(*helper.get_batches(X_train, y_train, self.batch_size)):
                logits = self.model.forward(x_batch)

                loss = self.loss_fn.forward(logits, y_batch)
                d_out = self.loss_fn.backward()

                self.model.backward(d_out)
                self.optimizer.step(self.model)

            log_msg = f"epoch {epoch}/{epochs} has passed"

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


