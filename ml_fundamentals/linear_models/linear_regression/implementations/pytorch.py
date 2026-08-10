import numpy as np
import torch
from torch import nn


class PyTorchLinearRegression:
    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-6) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps

        self.model = nn.Linear(in_features=in_features, out_features=1)
        self.model.to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 1000) -> None:
        self.model.train()

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y = torch.as_tensor(y, dtype=torch.float32, device=self.device).reshape(-1, 1)

        for epoch in range(1, max_iter + 1):
            self.optimizer.zero_grad()

            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)

            loss.backward()

            self.optimizer.step()

            grad = torch.cat([param.grad.flatten() for param in self.model.parameters()])
            if torch.linalg.vector_norm(grad) < self.eps:
                break

    def predict(self, X: np.ndarray) -> torch.Tensor:
        self.model.eval()
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            return self.model(X)
