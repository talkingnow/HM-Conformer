import torch
import torch.nn as nn

from .attention import Attention

class CLSBackend(nn.Module):
    def __init__(self, in_dim, hidden_dim, use_pooling=False, input_mean_std=False):
        super(CLSBackend, self).__init__()
        
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.silu = nn.SiLU()
        
        self.use_pooling = use_pooling
        if self.use_pooling:
            self.ASP = Attention("cls", in_dim, hidden_dim, input_mean_std=input_mean_std)
            self.fc_final = nn.Linear(hidden_dim * 2, hidden_dim)
            self.bn_final = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        assert len(x.size()) == 3, f'Input size error in pooling. Need 3, but get {len(x.size())}'
        
        cls = x[ :, 0, :]   # (B, H)
        cls = self.bn(self.silu(self.fc(cls)))
        
        if self.use_pooling:
            x = x[ :, 1:, :]    # (B, L, H)
            output = self.ASP(x)
            output = torch.cat((cls, output), dim=1)
            output = self.bn_final(self.fc_final(output))
        else:
            output = cls
        
        return output