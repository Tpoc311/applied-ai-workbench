import numpy as np
import torch
from torch import nn


class PyTorchLogisticRegression:
    def __init__(self, in_features: int, lr: float = 1e-3, eps: float = 1e-3) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eps = eps

        self.model: nn.Linear = nn.Linear(in_features=in_features, out_features=1).to(self.device)
        self.loss_fn: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.optimizer: torch.optim.SGD = torch.optim.SGD(self.model.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray, max_iter: int = 10000) -> np.ndarray:
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
        self.model.eval()

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.model(X)
            y_proba = torch.sigmoid(logits)

        return y_proba.flatten().cpu().numpy()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int64)

    @property
    def w(self) -> np.ndarray:
        weights = self.model.weight.detach().flatten()
        bias = self.model.bias.detach().flatten()

        return torch.cat((weights, bias)).reshape(-1, 1).cpu().numpy()
