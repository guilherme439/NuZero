""" 
Adapted from the Deepthinking repository.
"""
import math
import torch

from torch import nn

from .blocks import *

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, in_channels, policy_channels, block, num_blocks, width, recall=True, depth_wise_value=True, **kwargs):
        super().__init__()
        data_type = torch.float64

        self.recurrent = True

        self.recall = recall
        self.width = int(width)
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False, dtype=data_type)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                padding=1, stride=1, bias=False, dtype=data_type)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for b in range(num_blocks):
            recur_layers.append(block(self.width, self.width, stride=1, dtype=data_type))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)


        ## POLICY HEAD
                
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # Filter reduction before last layer
        # number of filters should be close to the dim of the output but not smaller
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, bias=False, dtype=data_type),
            nn.ReLU(),
            nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1, stride=1, bias=False, dtype=data_type),
            nn.ReLU(),
            nn.Conv2d(in_channels=width, out_channels=policy_filters, kernel_size=3, padding=1, stride=1, bias=False, dtype=data_type),
            nn.ReLU(),
            nn.Conv2d(in_channels=policy_filters, out_channels=policy_channels, kernel_size=3, padding=1, stride=1, bias=False, dtype=data_type),
        )


        ## VALUE HEAD
        if depth_wise_value:
            reduction_depth = 64
            self.value_head = nn.Sequential(
                nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding="same", stride=1, groups=width, bias=False, dtype=data_type),
                nn.Tanh(),
                nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding="same", stride=1, groups=width, bias=False, dtype=data_type),
                nn.Tanh(),
                nn.Conv2d(in_channels=width, out_channels=reduction_depth, kernel_size=3, padding="same", stride=1, bias=False, dtype=data_type),
                nn.Tanh(),
                nn.Conv2d(in_channels=reduction_depth, out_channels=1, kernel_size=3, padding="same", stride=1, bias=False, dtype=data_type),
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Tanh()
            )
        else:
            depth_conv_layers = [256, 64 , 8 , 1]

            value_head_layers = []
            current_depth = width
            for depth in depth_conv_layers:
                value_head_layers.append(nn.Conv2d(in_channels=current_depth, out_channels=depth, kernel_size=3,
                                                   padding="same", stride=1, bias=False, dtype=data_type))
                value_head_layers.append(nn.Tanh())
                current_depth = depth

            value_head_layers.append(nn.AdaptiveAvgPool3d(1))
            value_head_layers.append(nn.Flatten())
            value_head_layers.append(nn.Tanh())

            self.value_head = nn.Sequential(*value_head_layers)


        

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)

        policy_out = self.policy_head(interim_thought)
        value_out = self.value_head(interim_thought)
        out = (policy_out, value_out)

        return out


def dt_net_2d(in_channels, policy_channels, width, blocks, depth_wise_value=True):
    return DTNet(in_channels, policy_channels, BasicBlock2D, blocks, width=width, recall=False, depth_wise_value=depth_wise_value)


def dt_net_recall_2d(in_channels, policy_channels, width, blocks, depth_wise_value=True):
    return DTNet(in_channels, policy_channels, BasicBlock2D, blocks, width=width, recall=True, depth_wise_value=depth_wise_value)


