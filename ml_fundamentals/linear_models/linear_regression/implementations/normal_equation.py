import numpy as np
from numpy.linalg import inv


class NormalEquationRegression:
    """Linear regression solved analytically using the normal equation."""

    def __init__(self, in_features: int) -> None:
        """Initialize an unfitted linear regression model.

        :param in_features: Number of input features.
        """
        self.w: np.ndarray = np.zeros((in_features + 1, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the linear regression model.

        An intercept column is appended to the feature matrix before solving
        the normal equation.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Target values with shape ``(n_samples,)``.
        :return: Learned coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))
        y = y.reshape(-1, 1)

        self.w = inv(X.T @ X) @ X.T @ y

        return self.w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given samples.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :return: Predicted target values with shape ``(n_samples,)``.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(X @ self.w)
