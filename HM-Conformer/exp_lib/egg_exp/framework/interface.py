import copy
import torch
from typing import Optional
from abc import ABCMeta, abstractmethod

class Framework(metaclass=ABCMeta):
    def __init__(self):
        self.trainable_modules = {}
        self.device = 'cpu'
    
    @abstractmethod
    def __call__(self, x: torch.Tensor, *labels: Optional[torch.Tensor]):
        pass
    
    def use_distributed_data_parallel(self, device, find_unused_parameters=False):
        for key in self.trainable_modules.keys():
            self.trainable_modules[key].to(device) 

            if 0 < len(self.trainable_modules[key].state_dict().keys()):
                self.trainable_modules[key] = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.trainable_modules[key])
                self.trainable_modules[key] = torch.nn.parallel.DistributedDataParallel(
                        self.trainable_modules[key], device_ids=[device], find_unused_parameters=find_unused_parameters)

        self.device = device
        
    def get_parameters(self):
        params = []
        for model in self.trainable_modules.values():
            params += list(model.parameters())
        return params
    
    def copy_state_dict(self):
        output = {}
        for key, model in self.trainable_modules.items():
            if 0 < len(model.state_dict().keys()):
                output[key] = copy.deepcopy(model.state_dict())
            
        return output
        
    def load_state_dict(self, state_dict):
        for key, params in state_dict.items():
            self.trainable_modules[key].load_state_dict(params)

    def eval(self):
        for model in self.trainable_modules.values():
            model.eval()
            
    def train(self):
        for model in self.trainable_modules.values():
            model.train()
        
    def freeze(self):
        for model in self.trainable_modules.values():
            for param in model.parameters():
                param.requires_grad=False
            model.eval()