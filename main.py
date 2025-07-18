
import numpy as np
import logging
import helper

from losses import CrossEntropy
from mlp import MLP
from train import Trainer
from optimizers import SGD

logging.basicConfig(
    level=logging.INFO
)

test_data_path = 'cifar-10-batches-py/test_batch'
train_data_paths = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 5 + 1)]

X_raw_train, y_train, X_raw_test, y_test = helper.load_data(test_data_path, train_data_paths)
X_train, X_test = helper.normalize(X_raw_train), helper.normalize(X_raw_test)


if __name__ == '__main__':
    model = MLP(input_dim=3072, hidden_dim=64, output_dim=10)
    loss_fn = CrossEntropy()
    optimizer = SGD()

    epochs = 20
    idx_show_ep = 5

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_size=128
    )

    trainer.train(
        epochs=epochs,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test
    )

