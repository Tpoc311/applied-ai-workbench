import numpy as np
from numpy.linalg import inv


class NormalEquationRegression:
    def __init__(self):
        self.w = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))
        y = y.reshape(-1, 1)

        self.w = inv(X.T @ X) @ X.T @ y
        return self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(X @ self.w)
