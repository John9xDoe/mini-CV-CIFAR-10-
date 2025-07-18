import numpy as np

class CrossEntropy:
    def forward(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true
        batch_size = y_pred.shape[0]

        correct_probs = y_pred[np.arange(batch_size), y_true]
        loss = - np.mean(np.log(correct_probs + 1e-9)) # сглаживаение от log(0)
        return loss

    def backward(self):
        batch_size = self.y_pred.shape[0]
        grad = self.y_pred.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        grad /= batch_size
        return grad
