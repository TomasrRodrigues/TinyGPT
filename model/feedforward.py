import torch
from torch import nn

class FeedForward(nn.Module):
    """
    Simple feedforward neural network module used in the MLP block.
    Consists of two linear layers with a GELU activation in between.
    """
    def __init__(self, n_embd, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_embd)
        )

    def forward(self, x):
        return self.net(x)