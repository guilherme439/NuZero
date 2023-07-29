""" 
Adapted and simplified from the Deepthinking repository.
"""

from torch import nn
import torch.nn.functional as F

import hexagdly


class BasicBlock2D(nn.Module):
    """Basic residual block class 2D"""

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv1 = hexagdly.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                        kernel_size = 1, stride = 1, bias=False)

        self.conv2 = hexagdly.Conv2d(in_channels = in_channels, out_channels = out_channels,
                                        kernel_size = 1, stride = 1, bias=False)


    def forward(self, x):
        x = self.conv1(x) 
        out = self.conv2(x)
        return out