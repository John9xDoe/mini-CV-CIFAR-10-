import numpy as np
from helper import accuracy

class Tester:
    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def test(self, model):
        y_pred = model.predict(self.X_test)
        return accuracy(y_pred, self.y_test)

