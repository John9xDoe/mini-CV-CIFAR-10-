import itertools
import time

import numpy as np
import logging
import yaml

import helper
from losses import CrossEntropy
from metrics import accuracy
from mlp import MLP
from model_context import ExperimentContext
from optimizers import Momentum, SGD, RMSProp, Adam
from test import Tester
from visualisations import Visualizer


class Trainer:
    def __init__(self, model, loss_fn, optimizer, batch_size, lr):
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
        best_acc, cur_acc,  bad_epochs = 0, 0, 0

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
                    if bad_epochs != 0:
                        logging.info("bad epochs reset")
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    logging.info(f"bad epochs = {bad_epochs}")

                if bad_epochs >= patience:
                    logging.info(f"Early stop at epoch {epoch} - acc: {cur_acc:.4f}")
                    break

        result = {"accuracy": cur_acc}

        if timer:
            training_time = time.time() - start_time
            logging.info(f"training time: {training_time} s.")
            result["time"] = training_time

        if graph:
            Visualizer.plot_epochs_currency(epchs, accs)

        if save_res:
            ctx = ExperimentContext()
            with open(ctx.get_path('result.yaml'), 'w') as f:
                yaml.dump(result, f)
        return result


    def grid_search(self, params_trainer_grid, params_training_grid, params_model_grid):
        tester = Tester()

        params_trainer_combinations = list(itertools.product(*params_trainer_grid.values()))
        params_training_combinations = list(itertools.product(*params_training_grid.values()))
        params_models_combinations = list(itertools.product(*params_model_grid.values()))

        trainer_keys = list(params_trainer_grid.keys())
        training_keys = list(params_training_grid.keys())
        model_keys = list(params_model_grid.keys())

        results = {}
        total_combinations = len(params_trainer_combinations) * len(params_training_combinations) * len(params_models_combinations)

        idx = 0
        for i, trainer_combo in enumerate(params_trainer_combinations):
            args_trainer_dict = dict(zip(trainer_keys, trainer_combo))
            for j, training_combo in enumerate(params_training_combinations):
                args_training_dict = dict(zip(training_keys, training_combo))
                for k, model_combo in enumerate(params_models_combinations):
                    args_model_dict = dict(zip(model_keys, model_combo))

                    ExperimentContext._instance = None
                    ctx = ExperimentContext()

                    logging.info(f"Run model {idx}/{total_combinations}")
                    logging.info(f"trainer config: \n{args_trainer_dict}")
                    logging.info(f"training config: \n{args_training_dict}")
                    logging.info(f"model config: \n{args_model_dict}")

                    model = MLP(**args_model_dict)
                    trainer = Trainer(model, **args_trainer_dict)
                    trainer.train(**args_training_dict, epochs = float('inf'), timer=True, X_test=tester.X_test, y_test=tester.y_test, patience=10, save_res=True)

                    results[f"{ctx.timestamp}"] = tester.test(model)

                    model.save_model()
                    trainer.save_config()

                    idx += 1

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