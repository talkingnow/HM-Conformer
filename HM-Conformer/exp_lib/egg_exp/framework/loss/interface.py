import torch 
from abc import ABCMeta, abstractmethod

class Criterion(torch.nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        pass
