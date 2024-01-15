import random
import torch

class GaussianNoise:
    def __init__(self, p=0.5, mean=[0, 1], std=[1,20], snr=[10,20]):
        self.mean = mean
        self.std = std
        self.p = p
        self.snr = snr
        
    def __call__(self, x):
        if random.random() < self.p:
            m = random.uniform(self.mean[0], self.mean[1])
            s = random.uniform(self.std[0], self.std[1])
            n = torch.normal(m, s, x.size(), device=x.device, dtype=x.dtype)
            snr = random.uniform(self.snr[0], self.snr[1])
            x = x + n/snr
            return x
        else:
            return x