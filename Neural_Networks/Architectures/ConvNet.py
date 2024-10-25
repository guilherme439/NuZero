import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn

from .blocks import *

class ConvNet(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=1, num_filters=256, hex=True):

        super().__init__()
        self.recurrent = False

        self.kernel_size = kernel_size
        self.num_filters = num_filters

        num_layers = 6
        # General Module
        general_layer_list = []
        if hex:
            general_layer_list.append(hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=in_channels, out_channels=self.num_filters, bias=False))
        else:
            general_layer_list.append(nn.Conv2d(kernel_size=self.kernel_size, in_channels=in_channels, out_channels=self.num_filters, bias=False))
        general_layer_list.append(nn.ELU())
         
        for i in range(num_layers):
            if hex:
                general_layer_list.append(hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False))
            else:
                general_layer_list.append(nn.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False))
            general_layer_list.append(nn.ELU())
        
        
        self.general_module = nn.Sequential(*general_layer_list)
        
    
        # Policy Head
        self.policy_head = Conv_PolicyHead(self.num_filters, policy_channels, hex=hex)
        


        # Value Head
        self.value_head = Reduce_ValueHead(self.num_filters, hex=hex)


    def forward(self, x):

        x = self.general_module(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    



    