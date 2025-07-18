#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
import logging


logging.basicConfig(
    level=logging.INFO
)

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

def show_img(idx):
    img_flat = X_raw_train[idx]

    r = img_flat[0:1024].reshape(32,32)
    g = img_flat[1024:2048].reshape(32,32)
    b = img_flat[2048:].reshape(32,32)

    img = np.stack([r, g, b], axis=2)
    plt.imshow(img.astype('uint8'))
    #plt.axis('off')
    plt.title(f'ID: {idx}\nClass: {y_train[idx]}')
    plt.show()

def normalize(data):
    return data.astype(np.float32) / 255.0 # np explicitly casts it to float64 -> twice the memory

test_data_path = 'cifar-10-batches-py/test_batch'
train_data_paths = [f'cifar-10-batches-py/data_batch_{i}' for i in range(1, 5 + 1)]

X_raw_train, y_train, X_raw_test, y_test = load_data(test_data_path, train_data_paths)

print(X_raw_test.shape, len(y_test), X_raw_train.shape, len(y_train))

# Loss functions
def cross_entropy(y_pred, y_true):
    return -np.log(y_pred[y_true])

# Activation functions
def softmax(logits): # out layer
    logits -= np.max(logits) # explosion protection
    exp = np.exp(logits)
    return exp / np.sum(exp)

def relu(logits): # hide layers
    return np.maximum(logits, 0)

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

if __name__ == '__main__':
    idx = int(input("ID image: "))
    show_img(idx=idx)