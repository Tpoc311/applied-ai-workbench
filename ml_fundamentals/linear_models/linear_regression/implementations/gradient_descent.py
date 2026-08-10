import numpy as np
from numpy.linalg import norm


class GradientDescentRegression:
    def __init__(self, lr: float = 1e-3, max_iter: int = 10000, eps: float = 1e-7):
        self.learning_rate = lr
        self.max_iter = max_iter
        self.eps = eps

        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))
        y = y.reshape(-1, 1)

        i = 0
        self.w = np.array([0.0] * len(X[0])).reshape(-1, 1)
        while True:
            gradient = self.__gradient__(X, y)
            self.w = self.w - self.learning_rate * gradient

            if i == self.max_iter or norm(gradient) < self.eps:
                break

            i += 1

        return self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(X @ self.w)

    # def __mse__(self, X: np.ndarray, y: np.ndarray) -> np.floating:
    #     return 1 / len(X) * (norm(y - X @ self.w, 2) ** 2)

    def __gradient__(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 2 / len(X) * X.T @ (X @ self.w - y)
