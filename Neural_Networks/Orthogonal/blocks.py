""" 
Adapted and simplified from the Deepthinking repository.
"""
import torch
from torch import nn
import torch.nn.functional as F



class BasicBlock2D(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dtype=torch.float32):
        super().__init__()
        
        layers_list = []
        layers_list.append(nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                        kernel_size = 3, padding=1, stride = 1, bias=False, dtype=dtype))

        self.conv2 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                        kernel_size = 3, padding=1, stride = 1, bias=False, dtype=dtype)
        
        self.shortcut = nn.Sequential()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x += self.shortcut(x)
        out = F.relu(x)
        return out