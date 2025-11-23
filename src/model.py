import torch
import torch.nn as nn
import torch.nn.functional as F

rl = True
if rl:
    input = 42
else:
    input = 84


class ValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def forward(self, x):
        return self.net(x)
