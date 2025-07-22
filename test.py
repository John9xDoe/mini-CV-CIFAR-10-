import numpy as np
import metrics

class Tester:

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.init(*args, **kwargs)

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def test(self, model):
        y_pred = model.predict(self.X_test)
        return metrics.accuracy(y_pred, self.y_test)

