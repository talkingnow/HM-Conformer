import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Criterion

class CCE(Criterion):
    def __init__(self, embedding_size, num_class):
        super(Criterion, self).__init__()
        self.softmax_loss = nn.CrossEntropyLoss()
        self.fc1 = nn.Linear(embedding_size, num_class)
        self.bn1 = nn.BatchNorm1d(num_class)

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        if label is not None:
            loss = self.softmax_loss(x, label)
            return loss
        else:
            return x