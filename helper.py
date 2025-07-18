import logging
import pickle
import matplotlib.pyplot as plt

import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(test_data_path, train_data_paths):
    test_data = unpickle(test_data_path)
    train_data_dicts = []
    for path in train_data_paths:
        train_data_dicts.append(unpickle(path))
        logging.info(f'dict with path={path} unpickled')

    X_train = np.concatenate([d[b'data'] for d in train_data_dicts], axis=0)
    y_train = sum([d[b'labels'] for d in train_data_dicts], [])
    X_test, y_test = test_data[b'data'], test_data[b'labels']

    return X_train, y_train, X_test, y_test

def normalize(data):
    return data.astype(np.float32) / 255.0 # np explicitly casts it to float64 -> twice the memory

def get_batches(X, y, batch_size):
    return [X[i: i + batch_size] for i in range(0, len(X), batch_size)], [y[i: i + batch_size] for i in range(0, len(y), batch_size)]

# Weight initialization
def init_weights(mode, fan_in, fan_out):
    if mode == 'Xe': # Uniform
        limit = np.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-limit, limit, size=(fan_out, fan_in))
    elif mode == 'He': # Normal
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, size=(fan_out, fan_in))
    return ValueError("Unsupported mode")
# ? size

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true) * 100