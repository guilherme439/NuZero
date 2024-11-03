import math

import torch
from torch import nn
import torch.nn.functional as F

import hexagdly


class depthwise_conv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, bias=True, debug=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_layers_list = []
        for c in range(in_channels):
            conv = hexagdly.Conv2d(1, 1, kernel_size,stride, bias, debug)
            self.add_module("channel"+str(c)+"_conv", conv)
            self.conv_layers_list.append(conv)



    def forward(self, tensor):
        
        channel_list = torch.chunk(tensor, self.in_channels, dim=1)
        outputs_list = []

        for i in range(self.in_channels):
            input = channel_list[i]
            conv = self.conv_layers_list[i]
            output = conv(input)
            outputs_list.append(output)

        out = torch.cat(outputs_list, dim=1)   
        return out
    

##################################################################################################

    
