import logging
import pickle

import numpy as np

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