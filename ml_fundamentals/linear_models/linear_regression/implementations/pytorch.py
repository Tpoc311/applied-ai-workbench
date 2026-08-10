import numpy as np
import torch
from torch import nn


class PyTorchLinearRegression:
    """Linear regression implemented with ``nn.Linear`` and PyTorch autograd."""

    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        """Initialize the PyTorch linear regression model.

        :param in_features: Number of input features.
        :param lr: Learning rate used by the SGD optimizer.
        :param eps: Gradient-norm threshold used as the convergence criterion.
        """
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps

        self.model: nn.Linear = nn.Linear(in_features=in_features, out_features=1).to(self.device)
        self.loss_fn: nn.MSELoss = nn.MSELoss()
        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 10000) -> torch.Tensor:
        """Fit the linear regression model using full-batch gradient descent.

        The input arrays are converted to tensors and moved to the selected
        device. Gradients are computed automatically by PyTorch autograd and
        parameters are updated using SGD.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :param y: Target values with shape ``(n_samples,)``.
        :param max_iter: Maximum number of optimization iterations.
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

        # w = torch.tensor(
        #     [
        #         parameter.grad.flatten()
        #         for parameter in self.model.parameters()
        #         if parameter.grad is not None]
        # )
        # return w

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for the given samples.

        :param X: Feature matrix with shape ``(n_samples, n_features)``.
        :return: Predicted target values with shape ``(n_samples,)``.
        """
        self.model.eval()

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            y_pred = self.model(X)

        return y_pred.flatten().cpu().numpy()
