import numpy as np
from numpy.linalg import inv


class NormalEquationRegression:
    """Linear regression solved analytically using the normal equation."""

    def __init__(self) -> None:
        """Initialize an unfitted linear regression model."""
        self.w: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the linear regression model.

        An intercept column is appended to the feature matrix before solving
        the normal equation.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Target values with shape ``(n_samples,)``.
        :return: Learned coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        :raises np.linalg.LinAlgError: If the normal-equation matrix is
            singular and cannot be inverted.
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
        :raises RuntimeError: If the model has not been fitted yet.
        """
        if self.w is None:
            raise RuntimeError("The model must be fitted before prediction.")

        ones = np.ones((X.shape[0], 1))
        X = np.hstack((X, ones))

        return np.ravel(X @ self.w)
