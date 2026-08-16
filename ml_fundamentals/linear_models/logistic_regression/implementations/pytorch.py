import numpy as np
import torch
from torch import nn


class PyTorchLogisticRegression:
    """Logistic regression implemented with ``nn.Linear`` and PyTorch autograd."""

    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        """Initialize the PyTorch logistic regression model.
        :param in_features: Number of input features.
        :param lr: Learning rate used by the SGD optimizer.
        :param eps: Gradient-norm threshold used as the convergence criterion.
        """
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps

        self.model: nn.Linear = nn.Linear(in_features=in_features, out_features=1).to(self.device)
        self.loss_fn: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 10000) -> np.ndarray:
        """Fit the logistic regression model using full-batch gradient descent.
        The input arrays are converted to tensors and moved to the selected
        device. Gradients are computed automatically by PyTorch autograd and
        parameters are updated using SGD.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Binary target values with shape ``(n_samples,)``.
        :param max_iter: Maximum number of optimization iterations.
        :return: Learned coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        """
        self.model.train()

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)

        for _ in range(max_iter):
            self.optimizer.zero_grad()

            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            loss.backward()

            gradient = torch.cat([
                parameter.grad.flatten()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ])

            if torch.linalg.vector_norm(gradient) < self.eps:
                break

            self.optimizer.step()

        return self.w

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive-class probabilities for the given samples.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :return: Predicted probabilities with shape ``(n_samples,)``.
        """
        self.model.eval()

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X)
            y_proba = torch.sigmoid(logits)

        return y_proba.flatten().cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class labels for the given samples.
        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param threshold: Probability threshold used to convert probabilities into class labels.
        :return: Predicted binary labels with shape ``(n_samples,)``.
        """
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @property
    def w(self) -> np.ndarray:
        """Return learned model coefficients including the intercept.
        :return: Model coefficients with shape ``(n_features + 1, 1)``.
            The last coefficient represents the intercept.
        """
        weights = self.model.weight.detach().flatten()
        bias = self.model.bias.detach().flatten()

        return torch.cat((weights, bias)).reshape(-1, 1).cpu().numpy()
