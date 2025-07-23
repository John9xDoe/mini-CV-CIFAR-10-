import time

import logging
import yaml

import helper
from losses import CrossEntropy
from metrics import accuracy
from model_context import ExperimentContext
from optimizers import Momentum, SGD, RMSProp, Adam
from visualisations import Visualizer


class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size, lr=0.1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.lr = lr

    def train(self, epochs, X_train, y_train, X_test=None, y_test=None, log_interval=1, timer=False, graph=False, patience=None, min_delta=0.01, save_res=False):

        self.epochs = epochs
        if timer:
            start_time = time.time()

        accs, epchs = [], []
        best_acc, best_acc_epochs, best_acc_time, cur_acc, bad_epochs = 0, 0, 0, 0, 0

        epoch = 0
        while epoch < epochs:
            for x_batch, y_batch in zip(*helper.get_batches(X_train, y_train, self.batch_size)):
                logits = self.model.forward(x_batch)

                loss = self.loss_fn.forward(logits, y_batch)
                d_out = self.loss_fn.backward()

                self.model.backward(d_out)

                self.optimizer.step(self.model, self.lr)

            log_msg = f"epoch {epoch}/{epochs - 1} has passed"

            if epoch % log_interval == 0:
                y_pred = self.model.predict(X_test)
                log_msg += f": loss={loss:.4f}, accuracy={accuracy(y_pred, y_test):.2f}%"

            if graph:
                epchs.append(epoch + 1)
                accs.append(accuracy(y_pred, y_test))

            logging.info(log_msg)

            if patience:
                y_pred = self.model.predict(X_test)
                cur_acc = accuracy(y_pred, y_test)

                if cur_acc > best_acc + min_delta:
                    best_acc = cur_acc
                    best_acc_epochs = epoch
                    best_acc_time = time.time() - start_time

                    if bad_epochs != 0:
                        logging.info("bad epochs reset")
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    logging.info(f"bad epochs = {bad_epochs}")

                if bad_epochs >= patience:
                    logging.info(f"Early stop at epoch {epoch} - acc: {cur_acc:.4f}")
                    self.end_epoch = epoch
                    break

            epoch += 1

        result = {
            "accuracy": float(cur_acc), # float because cur_acc is np (bad printing for yaml)
            "best accuracy": float(best_acc),
            "best accuracy epochs": best_acc_epochs,
            "best accuracy time": best_acc_time
        }

        if timer:
            training_time = time.time() - start_time
            logging.info(f"training time: {training_time} s.")
            result["time"] = training_time

        if graph:
            Visualizer.plot_epochs_currency(epchs, accs, show=False)

        if save_res:
            ctx = ExperimentContext()
            with open(ctx.get_path('result.yaml'), 'w') as f:
                yaml.dump(result, f)
        return result

    def save_config(self):
        args = {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "epochs": self.end_epoch,
            "loss_fn": self.loss_fn.__class__.__name__,
            "optimizer": self.optimizer.__class__.__name__
        }

        ctx = ExperimentContext()

        with open (ctx.get_path('args.yaml'), 'w') as f:
            yaml.dump(args, f)

        logging.info(
            f"""config saved to '{ctx.get_path('args.yaml')}':
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
            "CrossEntropy": CrossEntropy
        }

        optimizers = {
            "SGD": SGD,
            "Momentum": Momentum,
            "RMSProp": RMSProp,
            "Adam": Adam
        }
        with open (f"experiments/model_{folder_id}/args.yaml", 'r') as f:
            args = yaml.safe_load(f)

        args["loss_fn"] = loss_fns[args["loss_fn"]]
        args["optimizer"] = optimizers[args["optimizer"]]

        logging.info(
            f"""config loaded from 'experiments/model_{folder_id}/args.yaml':
            batch_size: {args["batch_size"]}
            learning_rate: {args["lr"]}
            epochs: {args["epochs"]}
            loss_fn: {args["loss_fn"]}
            optimizer: {args["optimizer"]}
            """
        )

        return args