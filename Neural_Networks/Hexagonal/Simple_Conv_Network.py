import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn


class Simple_Conv_Network(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=1, num_filters=256):

        super(Simple_Conv_Network, self).__init__()


        self.kernel_size = kernel_size
        self.num_filters = num_filters


        # General Module
        self.general_module = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=in_channels, out_channels=self.num_filters, bias=False),
            nn.ELU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
            nn.ELU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
            nn.ELU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
            nn.ELU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
            nn.ELU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False)
        )

        
    
        # Policy Head
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # number of filters should be close to the dim of the output but not smaller (I think)

        self.policy_head = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=policy_filters, bias=False),
            nn.SiLU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=policy_filters, out_channels=policy_channels, bias=False),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )
        


        # Value Head
        processing_filters = int(self.num_filters/2)
        depth_of_final_stack = 1

        self.value_head = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=processing_filters, bias=False),
            nn.Hardtanh(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=processing_filters, out_channels=depth_of_final_stack, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Tanh()
        )


    def forward(self, x):

        x = self.general_module(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    



    