import torch
import torch.nn as nn
import torch.nn.functional as F


class ChessNet(nn.Module):
    """
    A neural network for evaluating chess board positions.

    Architecture:
    - 3 convolutional layers to extract spatial and positional features from the board.
    - 1 fully connected layer to map the features to a final scalar output.
    - ReLU activations and a final sigmoid to bound the output between 0 and 1.
    """

    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)
        return torch.sigmoid(self.fc1(x))
