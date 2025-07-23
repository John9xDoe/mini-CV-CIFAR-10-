import itertools
import logging

from activations import ReLu
from losses import CrossEntropy
from mlp import MLP
from model_context import ExperimentContext
from optimizers import SGD, Momentum, RMSProp, Adam
from test import Tester
from train import Trainer


class ExperimentRunner:
    def __init__(self, params_trainer_grid=None, params_model_grid=None):
        self.params_trainer_grid = params_trainer_grid
        self.params_model_grid = params_model_grid

        self.INPUT_DIM = 3072
        self.OUTPUT_DIM = 10

        if params_model_grid is None:
            self.params_model_grid = {
                "activation_class": [ReLu],
                "hidden_dim": [128]
            }

        if params_trainer_grid is None:
            self.params_trainer_grid = {
                "loss_fn": [CrossEntropy],
                "optimizer": [SGD],
                "batch_size": [128],
                "lr": [0.1, 0.01, 0.001]
            }

    def grid_search(self, X_train, y_train):
        tester = Tester.get_instance()

        params_trainer_combinations = list(itertools.product(*self.params_trainer_grid.values()))
        params_models_combinations = list(itertools.product(*self.params_model_grid.values()))

        trainer_keys = list(self.params_trainer_grid.keys())
        model_keys = list(self.params_model_grid.keys())

        results = {}
        total_combinations = len(params_trainer_combinations) * len(params_models_combinations)

        idx = 0
        for i, trainer_combo in enumerate(params_trainer_combinations):
            args_trainer_dict = dict(zip(trainer_keys, trainer_combo))
            args_trainer_dict["loss_fn"] = args_trainer_dict["loss_fn"]()

            for j, model_combo in enumerate(params_models_combinations):
                args_model_dict = dict(zip(model_keys, model_combo))

                ExperimentContext._instance = None
                ctx = ExperimentContext()

                logging.info(f"Run model {idx}/{total_combinations}")
                logging.info(f"trainer config: \n{args_trainer_dict}")
                logging.info(f"model config: \n{args_model_dict}")

                model = MLP(**args_model_dict, input_dim=self.INPUT_DIM, output_dim=self.OUTPUT_DIM)
                optimizer_class = args_trainer_dict["optimizer"]
                optimizer = optimizer_class(model, lr=args_trainer_dict["lr"])

                args_trainer_dict_copy = args_trainer_dict.copy()
                del args_trainer_dict_copy["optimizer"]

                trainer = Trainer(model, **args_trainer_dict_copy, optimizer=optimizer)
                trainer.train(
                    epochs = float('inf'),
                    X_train=X_train,
                    y_train=y_train,
                    timer=True,
                    X_test=tester.X_test,
                    y_test=tester.y_test,
                    patience=15,
                    save_res=True,
                    graph=True
                )

                results[f"{ctx.timestamp}_{idx}"] = tester.test(model)

                model.save_model()
                trainer.save_config()

                idx += 1
        return max(results.items(), key=lambda x: x[1])