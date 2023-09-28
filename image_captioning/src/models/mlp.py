import torch
import torch.nn as nn
from typing import Tuple

"""
Code adapted from: https://github.com/rmokady/CLIP_prefix_caption/blob/main/train.py
"""

class MLP(nn.Module):

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)