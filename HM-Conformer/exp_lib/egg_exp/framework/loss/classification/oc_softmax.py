import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Criterion

class OCSoftmax(Criterion):
    def __init__(self, embedding_size, num_class=1, feat_dim=2, r_real=0.9, r_fake=0.2, alpha=20.0, use_cls_weight=False):
        super(Criterion, self).__init__()
        self.embedding_size = embedding_size
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.use_cls_weight = use_cls_weight
        
        self.weight = nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, 0.25)
        self.weight.data.renorm_(2,1,1e-5).mul_(1e5)
        
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor, label=None) -> torch.Tensor:
            
        w = F.normalize(self.weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        
        scores = x @ w.transpose(0, 1)
        if label is None:
            scores = 1 - scores # Caution
            return scores

        scores[label == 0] = self.r_real - scores[label == 0]
        scores[label == 1] = scores[label == 1] - self.r_fake

        loss = self.softplus(self.alpha * scores)
        if self.use_cls_weight:
            loss[label == 0] = loss[label == 0] * 1.8
            loss[label == 1] = loss[label == 1] * 0.2
        loss = loss.mean()
        return loss