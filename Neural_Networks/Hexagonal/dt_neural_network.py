""" 
Adapted and simplified from the Deepthinking repository.
"""
import math
import torch
import hexagdly

from torch import nn

from .blocks import BasicBlock2D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, in_channels, policy_channels, block, num_blocks, width, recall=True, **kwargs):
        super().__init__()
        self.recurrent = True
        
        self.recall = recall
        self.width = int(width)
        proj_conv = hexagdly.Conv2d(in_channels, width, kernel_size=1,
                              stride=1, bias=False)

        conv_recall = hexagdly.Conv2d(width + in_channels, width, kernel_size=1,
                                stride=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for b in range(num_blocks):
            recur_layers.append(block(self.width, self.width, stride=1))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)


        ## POLICY HEAD
        # number of filters should be close to the dim of the output but not smaller
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) 

        self.policy_head = nn.Sequential(
            hexagdly.Conv2d(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            hexagdly.Conv2d(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            hexagdly.Conv2d(in_channels=width, out_channels=policy_filters, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            hexagdly.Conv2d(in_channels=policy_filters, out_channels=policy_channels, kernel_size=1, stride=1, bias=False),
        )


        ## VALUE HEAD

        depth_conv_layers = [256, 64 , 8 , 1]

        value_head_layers = []
        current_depth = width
        for depth in depth_conv_layers:
            value_head_layers.append(hexagdly.Conv2d(in_channels=current_depth, out_channels=depth, kernel_size=1, stride=1, bias=False))
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


def dt_net_2d(in_channels, policy_channels, width, num_blocks):
    return DTNet(in_channels, policy_channels, BasicBlock, num_blocks, width=width, recall=False)


def dt_net_recall_2d(in_channels, policy_channels, width, num_blocks):
    return DTNet(in_channels, policy_channels, BasicBlock, num_blocks, width=width, recall=True)
