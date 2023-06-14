""" dt_net_2d.py
    DeepThinking network 2D.
    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.
    Developed for DeepThinking project
    October 2021
"""
import math
import torch
from torch import nn

from .blocks import BasicBlock2D as BasicBlock

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class DTNet(nn.Module):
    """DeepThinking Network 2D model class"""

    def __init__(self, game, block, num_blocks, width, recall=True, group_norm=False, **kwargs):
        super().__init__()

        action_space_shape = game.get_action_space_shape()
        total_action_planes = list(action_space_shape)[0]
        input_shape = game.state_shape()
        in_channels = input_shape[0]

        self.recall = recall
        self.width = int(width)
        self.group_norm = group_norm
        proj_conv = nn.Conv2d(in_channels, width, kernel_size=3,
                              stride=1, padding=1, bias=False)

        conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3,
                                stride=1, padding=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)

        '''
        head_conv1 = nn.Conv2d(width, 32, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv2 = nn.Conv2d(32, 8, kernel_size=3,
                               stride=1, padding=1, bias=False)
        head_conv3 = nn.Conv2d(8, 2, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.head = nn.Sequential(head_conv1, nn.ReLU(),
                                  head_conv2, nn.ReLU(),
                                  head_conv3)
        '''

        # number of filters should be close to the dim of the output but not smaller (I think)
        policy_filters = int(math.pow(2, math.ceil(math.log(total_action_planes, 2)))) 
        
        self.policy_head = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=policy_filters, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=policy_filters),
            nn.ReLU(),
            nn.Conv2d(in_channels=policy_filters, out_channels=total_action_planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Flatten(), # there in no softmax for a 3D policy (that I am aware), so we need to flatten and unflatten the policy to apply softmax
            nn.Softmax(dim=1),
            nn.Unflatten(1, action_space_shape)
        )


        # Value Head
        depth_of_first_stack = 32
        depth_of_final_stack = 1

        self.value_head = nn.Sequential(
            nn.Conv2d(in_channels=width, out_channels=depth_of_first_stack, kernel_size=3,stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depth_of_first_stack),
            nn.Hardtanh(),
            nn.Conv2d(in_channels=depth_of_first_stack, out_channels=depth_of_final_stack, kernel_size=3,stride=1, padding=1, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd, group_norm=self.group_norm))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        all_outputs = []

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            policy_out = self.policy_head(interim_thought)
            value_out = self.value_head(interim_thought)
            out = (policy_out, value_out)
            all_outputs.append(out)

        #if self.training:
            #return out, interim_thought

        return all_outputs[-1]


def dt_net_2d(game, width, **kwargs):
    return DTNet(game ,BasicBlock, [2], width=width, recall=False)


def dt_net_recall_2d(game, width, **kwargs):
    return DTNet(game, BasicBlock, [2], width=width, recall=True)


def dt_net_gn_2d(game, width, **kwargs):
    return DTNet(game, BasicBlock, [2], width=width, recall=False, group_norm=True)


def dt_net_recall_gn_2d(game, width, **kwargs):
    return DTNet(game, BasicBlock, [2], width=width, recall=True, group_norm=True)