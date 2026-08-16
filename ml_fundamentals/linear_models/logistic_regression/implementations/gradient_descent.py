import numpy as np
from numpy.linalg import norm


class GradientDescentLogisticRegression:
    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        self.learning_rate = lr
        self.eps = eps

        self.w: np.ndarray = np.zeros((in_features + 1, 1))

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))
        y = y.reshape(-1, 1)

        for _ in range(max_iter):
            gradient = self._gradient(X, y)

            if norm(gradient) < self.eps:
                break

            self.w -= self.learning_rate * gradient

        return self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(self._sigmoid(X @ self.w))

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self._sigmoid(X @ self.w)
        return X.T @ (p - y) / len(X)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def bce_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pos_class = y_true * np.log(y_pred)
        neg_class = (1 - y_true) * np.log(1 - y_pred)
        return -np.mean(pos_class + neg_class)
