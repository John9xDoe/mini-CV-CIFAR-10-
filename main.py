
import numpy as np
import logging
import helper
from activations import ReLu

from losses import CrossEntropy
from mlp import MLP
from train import Trainer
from optimizers import SGD
from load_data import DataLoader

logging.basicConfig(
    level=logging.INFO
)

test_data_path = 'cifar-10-batches-py/test_batch'
train_data_paths = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 5 + 1)]

def start(mode):
    X_raw_train, y_train, X_raw_test, y_test = DataLoader.load_data(train_data_paths, test_data_path)
    X_train, X_test = helper.normalize(X_raw_train), helper.normalize(X_raw_test)  # explosion protection

    model = MLP(input_dim=3072, hidden_dim=128, output_dim=10, activation_class=ReLu, lr=0.001)
    loss_fn = CrossEntropy()
    optimizer = SGD()

    if mode == 'train':

        epochs = 100
        #idx_show_ep = 5

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
            y_test=y_test,
            timer=True,
            graph=True
        )

        model.save_model(epochs)

    elif mode == 'load_model':
        model_name = input('folder_id: ')
        model = MLP.load_model(model_name, input_dim=3072, output_dim=10)

if __name__ == '__main__':

    #start(mode='load_model')
    start(mode='train')





