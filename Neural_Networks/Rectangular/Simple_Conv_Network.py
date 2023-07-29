import time
import os
import math
import numpy as np
import torch
from torch import nn



class Simple_Conv_Network(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=(3,3), num_filters=256):

        super(Simple_Conv_Network, self).__init__()


        self.kernel_size = kernel_size
        self.num_filters = num_filters


        # General Module
        self.general_module = nn.Sequential(
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=in_channels, out_channels=64),
            nn.ELU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=self.num_filters),
            nn.ELU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=self.num_filters),
            nn.ELU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=self.num_filters),
            nn.ELU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=self.num_filters),
            nn.ELU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=self.num_filters)
        )

        
    
        # Policy Head
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # number of filters should be close to the dim of the output but not smaller (I think)

        self.policy_head = nn.Sequential(
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=policy_filters),
            nn.SiLU(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=policy_filters, out_channels=policy_channels),
            nn.Flatten(),
            nn.Softmax(dim=1)
        )
        


        # Value Head
        processing_filters = int(self.num_filters/2)
        depth_of_final_stack = 1

        self.value_head = nn.Sequential(
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=self.num_filters, out_channels=processing_filters),
            nn.Hardtanh(),
            nn.Conv2d(kernel_size=self.kernel_size, padding="same", in_channels=processing_filters, out_channels=depth_of_final_stack),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Tanh()
        )


    def forward(self, x):

        x = self.general_module(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    



    