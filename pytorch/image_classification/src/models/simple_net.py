import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    Simple convolutional neural network for CIFAR-10 classification.

    Architecture consists of two convolutional layers with max pooling,
    followed by three fully connected layers. Outputs raw logits.
    """

    def __init__(self):
        """
        Initialize network layers and set up the computational graph.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Execute the forward pass of the network.

        :param x: Input batch of CIFAR-10 images with shape (B, 3, 32, 32).
        :return: Raw classification logits with shape (B, 10).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
