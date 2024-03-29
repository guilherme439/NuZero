""" 
Adapted from the Deepthinking repository.
"""
import math
import torch
import hexagdly

from torch import nn

from .blocks import *

# Ignore statemenst for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702)
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914


class RecurrentNet(nn.Module):

    def __init__(self, in_channels, policy_channels, width, num_blocks, recall=True, policy_head="conv", value_head="reduce", value_activation="relu", hex=True):
        super().__init__()
        self.recurrent = True
        
        self.recall = recall
        self.width = int(width)
        if hex:
            proj_conv = hexagdly.Conv2d(in_channels, width, kernel_size=1, stride=1, bias=False)
        else:
            proj_conv = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, bias=False)

        if hex:
            conv_recall = hexagdly.Conv2d(width + in_channels, width, kernel_size=1, stride=1, bias=False)
        else:
            conv_recall = nn.Conv2d(width + in_channels, width, kernel_size=3, stride=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for b in range(num_blocks):
            recur_layers.append(BasicBlock(self.width, hex=hex))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_module = nn.Sequential(*recur_layers)


        ## POLICY HEAD
        match policy_head:
            case "conv":
                self.policy_head = Conv_PolicyHead(width, policy_channels, hex=hex)
            case _:
                print("Unknown choice")
                exit()
        

        ## VALUE HEAD
        match value_head:
            case "reduce":
                self.value_head = Reduce_ValueHead(width, activation=value_activation, hex=hex)
            case "depth":
                self.value_head = Depth_ValueHead(width, activation=value_activation, hex=hex)
            case "combined":
                self.value_head = Combined_ValueHead(width, activation=value_activation, hex=hex)
            case "separable":
                self.value_head = Separable_ValueHead(width, activation=value_activation, hex=hex)
            case "reverse":
                self.value_head = Reverse_ValueHead(width, activation=value_activation, hex=hex)
            case "rawsep":
                self.value_head = RawSeparable_ValueHead(width, activation=value_activation, hex=hex)
            case "strange":
                self.value_head = Strange_ValueHead(width, activation=value_activation, hex=hex)
            case "dense":
                self.value_head = Dense_ValueHead(width)
            case _:
                print("Unknown choice")
                exit()

        


    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought


        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_module(interim_thought)
        

        policy_out = self.policy_head(interim_thought)
        value_out = self.value_head(interim_thought)
        out = (policy_out, value_out)
        
        return out, interim_thought




