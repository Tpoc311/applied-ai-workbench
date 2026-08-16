import numpy as np
from numpy.linalg import norm


class GradientDescentLogisticRegression:
    """Logistic regression optimized with full-batch gradient descent."""

    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        """Initialize the gradient descent optimizer.
        :param in_features: Number of input features.
        :param lr: Learning rate used for parameter updates.
        :param eps: Gradient-norm threshold used as the convergence criterion.
        """
        self.learning_rate = lr
        self.eps = eps

        self.w: np.ndarray = np.zeros((in_features + 1, 1), dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 10000) -> np.ndarray:
        """Fit the logistic regression model using gradient descent.
        An intercept column is appended to the feature matrix. Parameters are updated until
        either the gradient norm falls below the convergence threshold or the maximum number of iterations is reached.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Binary target values with shape ``(n_samples,)``.
        :param max_iter: Maximum number of optimization iterations.
        :return: Learned coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.hstack((X, ones))
        y = y.reshape(-1, 1)

        for _ in range(max_iter):
            gradient = self._gradient(X, y)

            if norm(gradient) < self.eps:
                break

            self.w -= self.learning_rate * gradient

        return self.w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for the given samples.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :return: Predicted probabilities with shape ``(n_samples,)``.
        """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        X = np.hstack((X, ones))

        return np.ravel(self._sigmoid(X @ self.w))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels for the given samples.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param threshold: Probability threshold used to convert probabilities into class labels.
        :return: Predicted binary labels with shape ``(n_samples,)``.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    def _gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate the gradient of binary cross-entropy.
        :param X: Feature matrix including the intercept column.
        :param y: Binary target column vector.
        :return: Gradient with respect to all model parameters.
        """
        p = self._sigmoid(X @ self.w)

        return X.T @ (p - y) / len(X)

    def _sigmoid(self, X: np.ndarray) -> np.ndarray:
        """Apply the sigmoid function element-wise.
        :param X: Input values.
        :return: Sigmoid-transformed values with the same shape as the input.
        """
        return 1 / (1 + np.exp(-X))
