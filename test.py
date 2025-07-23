import metrics

class Tester:

    _instance = None

    @classmethod
    def get_instance(cls, X_test=None, y_test=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance = cls(X_test, y_test)
        return cls._instance

    def __init__(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def test(self, model):
        y_pred = model.predict(self.X_test)
        return metrics.accuracy(y_pred, self.y_test)

