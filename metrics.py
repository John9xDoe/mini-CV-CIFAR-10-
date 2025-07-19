import numpy as np

def accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true) * 100

