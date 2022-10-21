import math
import torch
from torch import nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, n_arms):
        super(PolicyNet, self).__init__()
        n_middle = 3 * n_arms
        self.fc1 = nn.Linear(n_arms, n_middle)
        self.fcout = nn.Linear(n_middle, n_arms)

        # layers = 2
        layers = int(math.ceil(n_arms / 2))  # todo
        self.blocks = nn.ModuleList([nn.Linear(n_middle, n_middle) for _ in range(layers)])
        # todo: try BN maybe useful for large K?

    def forward(self, x):
        # resnet-like network
        h = F.relu(self.fc1(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        output = F.softmax(self.fcout(h), dim=1)

        return output

