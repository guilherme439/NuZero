import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn




class ResNet(nn.Module):

    def __init__(self, in_channels, policy_channels, num_blocks=4, kernel_size=1, num_filters=256):

        super(ResNet, self).__init__()

        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.first_block_list = []
        self.second_block_list = []
        self.all_blocks_list = []

        # General Module
        self.input_block = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=in_channels, out_channels=self.num_filters, bias=False),
            nn.BatchNorm2d(num_features=self.num_filters),
            nn.ReLU()
        )

        for block in range(self.num_blocks):
            first_block = nn.Sequential(
                hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
                nn.BatchNorm2d(num_features=self.num_filters),
                nn.ReLU(),
            )
            self.first_block_list.append(first_block)

            second_block = nn.Sequential(
                hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=self.num_filters, bias=False),
                nn.BatchNorm2d(num_features=self.num_filters),
            )

            self.second_block_list.append(second_block)

            self.all_blocks_list.append(first_block)
            self.all_blocks_list.append(second_block)


        self.residual_blocks = nn.Sequential(*self.all_blocks_list)

        # Policy Head
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # number of filters should be close to the dim of the output but not smaller (I think)
        
        self.policy_head = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=policy_filters, bias=False),
            nn.BatchNorm2d(num_features=policy_filters),
            nn.ReLU(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=policy_filters, out_channels=policy_channels, bias=False),
        )


        # Value Head
        depth_of_first_stack = 32
        depth_of_final_stack = 1

        self.value_head = nn.Sequential(
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=self.num_filters, out_channels=depth_of_first_stack, bias=False),
            nn.BatchNorm2d(num_features=depth_of_first_stack),
            nn.Hardtanh(),
            hexagdly.Conv2d(kernel_size=self.kernel_size, in_channels=depth_of_first_stack, out_channels=depth_of_final_stack, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Tanh()
        )
    


    def forward(self, x):

        processed_input = self.input_block(x)

        last_block_out = processed_input
        for block_index in range(self.num_blocks):
            x = self.first_block_list[block_index](last_block_out)
            x = self.second_block_list[block_index](x)

            skip_connection = x + last_block_out

            re_lu = nn.ReLU()
            last_block_out = re_lu(skip_connection)

        
        policy = self.policy_head(last_block_out)
        value = self.value_head(last_block_out)
        
        return policy, value
    


    