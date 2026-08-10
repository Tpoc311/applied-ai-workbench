import numpy as np
from numpy.linalg import norm


class GradientDescentRegression:
    """Linear regression optimized with full-batch gradient descent."""

    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        """Initialize the gradient descent optimizer.

        :param in_features: Number of input features.
        :param lr: Learning rate used for parameter updates.
        :param eps: Gradient-norm threshold used as the convergence criterion.
        """
        self.learning_rate = lr
        self.eps = eps

        self.w: np.ndarray = np.zeros((in_features + 1, 1))

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 10000) -> np.ndarray:
        """Fit the linear regression model using gradient descent.

        An intercept column is appended to the feature matrix. Parameters are updated until
        either the gradient norm falls below the convergence threshold or the maximum number of iterations is reached.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Target values with shape ``(n_samples,)``.
        :param max_iter: Maximum number of optimization iterations.
        :return: Learned coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        """
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
        """Predict target values for the given samples.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :return: Predicted target values with shape ``(n_samples,)``.
        """
        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(X @ self.w)

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the gradient of the mean squared error.

        :param X: Feature matrix including the intercept column.
        :param y: Target column vector.
        :return: Gradient with respect to all model parameters.
        """
        return 2 / len(X) * X.T @ (X @ self.w - y)
