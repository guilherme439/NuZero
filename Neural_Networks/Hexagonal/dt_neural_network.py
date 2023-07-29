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


        self.recall = recall
        self.width = int(width)
        proj_conv = hexagdly.Conv2d(in_channels, width, kernel_size=1,
                              stride=1, bias=False)

        conv_recall = hexagdly.Conv2d(width + in_channels, width, kernel_size=1,
                                stride=1, bias=False)

        recur_layers = []
        if recall:
            recur_layers.append(conv_recall)

        for i in range(len(num_blocks)):
            recur_layers.append(self._make_layer(block, width, num_blocks[i], stride=1))

        
        self.projection = nn.Sequential(proj_conv, nn.ReLU())
        self.recur_block = nn.Sequential(*recur_layers)


        ## POLICY HEAD
        # number of filters should be close to the dim of the output but not smaller (I think)
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) 
        
        self.policy_head = nn.Sequential(
            hexagdly.Conv2d(in_channels=width, out_channels=policy_filters, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),
            hexagdly.Conv2d(in_channels=policy_filters, out_channels=policy_channels, kernel_size=1, stride=1, bias=False),
            nn.Flatten(), # there is no softmax function for a 3D policy in pytorch (that I am aware), so we need to flatten the policy to apply softmax
            nn.Softmax(dim=1)
        )


        ## VALUE HEAD
        depth_of_first_stack = 32
        depth_of_final_stack = 1

        self.value_head = nn.Sequential(
            hexagdly.Conv2d(in_channels=width, out_channels=depth_of_first_stack, kernel_size=1, stride=1, bias=False),
            nn.Hardtanh(),
            hexagdly.Conv2d(in_channels=depth_of_first_stack, out_channels=depth_of_final_stack, kernel_size=1, stride=1, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Tanh()
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.width, planes, strd))
            self.width = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        #all_outputs = []
        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            policy_out = self.policy_head(interim_thought)
            value_out = self.value_head(interim_thought)
            out = (policy_out, value_out)
            #all_outputs.append(out)

        #if self.training:
            #return out, interim_thought

        return out


def dt_net_2d(in_channels, policy_channels, width, **kwargs):
    return DTNet(in_channels, policy_channels, BasicBlock, [2], width=width, recall=False)


def dt_net_recall_2d(in_channels, policy_channels, width, **kwargs):
    return DTNet(in_channels, policy_channels, BasicBlock, [2], width=width, recall=True)
