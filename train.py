import time

import numpy as np
import logging

import helper
from linear import Linear

class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size

    def train(self, epochs, X_train, y_train, X_test=None, y_test=None, log_interval=1, timer=False):
        if timer:
            start_time = time.time()

        for epoch in range(epochs + 1):
            for x_batch, y_batch in zip(*helper.get_batches(X_train, y_train, 128)):
                logits = self.model.forward(x_batch)

                loss = self.loss_fn.forward(logits, y_batch)
                d_out = self.loss_fn.backward()

                self.model.backward(d_out)
                self.optimizer.step(self.model)

            log_msg = f"epoch {epoch}/{epochs} has passed"

            if epoch % log_interval == 0:
                y_pred = self.model.predict(X_test)
                log_msg += f": loss={loss:.4f}, accuracy={evaluate(y_pred, y_test):.2f}%"

            logging.info(log_msg)

        if timer:
            logging.info(f"training time: {time.time() - start_time} s.")


