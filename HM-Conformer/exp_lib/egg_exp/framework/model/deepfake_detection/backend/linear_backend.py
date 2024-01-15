import torch
import torch.nn as nn

class LinearBackend(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(LinearBackend, self).__init__()
        
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
                
    def forward(self, x):
        assert len(x.size()) == 2, f'Input size error in pooling. Need 2, but get {len(x.size())}'
        output = self.bn(self.fc(x))
        
        return output