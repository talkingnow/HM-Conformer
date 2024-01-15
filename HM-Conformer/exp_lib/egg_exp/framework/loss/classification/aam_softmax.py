import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import Criterion

class AAMSoftmax(Criterion):
    def __init__(self, embedding_size, num_class, margin, scale, class_weight=None, topk_panelty=None):
        super(AAMSoftmax, self).__init__()
        
        self.scale = scale
        
        # fc
        self.fc = torch.nn.Parameter(torch.FloatTensor(num_class, embedding_size), requires_grad=True)
        nn.init.xavier_normal_(self.fc, gain=1)
        
        # CE
        if class_weight is not None:
            if type(class_weight) is list:
                class_weight = torch.tensor(class_weight)
            self.ce = nn.CrossEntropyLoss(weight=class_weight)
        else:
            self.ce = nn.CrossEntropyLoss()
        
        # positive_margine
        self.cos_pos_m = math.cos(margin)
        self.sin_pos_m = math.sin(margin)

        # topk panalty
        self.topk = None
        if topk_panelty is not None:
            self.topk, topk_margin = topk_panelty
            self.cos_neg_m = math.cos(topk_margin)
            self.sin_neg_m = math.sin(topk_margin)
            
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.fc))
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        
        # cos(theta + m) -> utilized angle addition and subtraction formulas
        phi = cosine * self.cos_pos_m - sine * self.sin_pos_m
    
        # cos(theta - m)
        if self.topk is not None:
            panalty_cos = cosine * self.cos_neg_m + sine * self.sin_neg_m
        
        # one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        
        if self.topk is not None:
            # except self
            idx_map = torch.ones(cosine.size(), device=cosine.device)
            idx_map.scatter_(1, label.view(-1, 1), -2)
            negative_idx = cosine * idx_map
            
            # select topK index (panalty_idx)
            K = self.topk
            topk = negative_idx.topk(K, dim=1)[1]
            panalty_idx = torch.zeros_like(negative_idx)           
            panalty_idx.scatter_(1, topk.view(-1, K), 1)
            
            # calculate loss
            remainder = 1 - one_hot - panalty_idx
            output = (one_hot * phi) + (panalty_idx * panalty_cos) + (remainder * cosine)
            output = output * self.scale
            
            loss = self.ce(output, label)
            
            return loss
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.scale
            
            loss = self.ce(output, label)
            
            return loss