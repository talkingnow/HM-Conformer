import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Criterion

class P2SGrad(Criterion):
    def __init__(self, embedding_size, num_class=2):
        super(Criterion, self).__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        
        self.mse = nn.MSELoss()
        
    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        scores = x @ w.transpose(0, 1)
        if label is None:
            return scores
        
        with torch.no_grad():
            idx = torch.zeros_like(scores)
            idx.scatter_(1, label.data.view(-1, 1), 1)
            
        loss = self.mse(scores, idx)
        return loss