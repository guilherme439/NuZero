import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn

from .blocks import * 


class ResNet(nn.Module):

    def __init__(self, in_channels, policy_channels, kernel_size=1, num_filters=256, num_blocks=4, batch_norm=False, policy_head="conv", value_head="reduce", value_activation="relu", hex=True):

        super().__init__()
        self.recurrent = False

        # Input Module
        input_layers = []
        if hex:
            input_layers.append(hexagdly.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=num_filters, bias=False))
        else:
            input_layers.append(nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=num_filters, bias=False))
        if batch_norm:
            input_layers.append(nn.BatchNorm2d(num_features=num_filters))
        input_layers.append(nn.ReLU())

        self.input_block = nn.Sequential(*input_layers)


        # Processing module
        residual_blocks_list = []
        for b in range(num_blocks):
            residual_blocks_list.append(BasicBlock(num_filters, batch_norm=batch_norm, hex=hex))

        self.residual_blocks = nn.Sequential(*residual_blocks_list)


        # Output Module
        ## POLICY HEAD
        match policy_head:
            case "conv":
                self.policy_head = Conv_PolicyHead(num_filters, policy_channels, batch_norm=batch_norm, hex=hex)
            case _:
                print("Option not available for ResNet")
                exit()
        
        ## VALUE HEAD
        match value_head:
            case "reduce":
                self.value_head = Reduce_ValueHead(num_filters, activation="relu", batch_norm=batch_norm, hex=hex)
            case "dense":
                self.value_head = Dense_ValueHead(num_filters, batch_norm=batch_norm, hex=hex)
            case _:
                print("Option not available for ResNet")
                exit()
    


    def forward(self, x):
        projection = self.input_block(x)

        processed_data = self.residual_blocks(projection)

        policy = self.policy_head(processed_data)
        value = self.value_head(processed_data)
        
        return policy, value
    


    