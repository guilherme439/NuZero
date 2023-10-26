
import math

import torch
from torch import nn
import torch.nn.functional as F

import hexagdly


class BasicBlock(nn.Module):

    def __init__(self, channels, batch_norm=False):
        super().__init__()
        
        before_shortcut_layers = []

        before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        if batch_norm:
            before_shortcut_layers.append(nn.BatchNorm2d(num_features=channels))
        before_shortcut_layers.append(nn.ReLU())
        before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        
        self.before_shortcut = nn.Sequential(*before_shortcut_layers)
        self.shortcut = nn.Sequential()



    def forward(self, x):
        out = self.before_shortcut(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

##################################################################################################

class Reduce_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        depth_conv_layers = [64 , 8 , 1]

        value_head_layers = []
        current_depth = width
        for depth in depth_conv_layers:
            conv = hexagdly.Conv2d(in_channels=current_depth, out_channels=depth, kernel_size=1, stride=1, bias=False)
            value_head_layers.append(conv)
            if depth != 1:
                if batch_norm:
                    value_head_layers.append(nn.BatchNorm2d(num_features=depth))

                if activation == "tanh":
                    value_head_layers.append(nn.Tanh())
                elif activation == "relu":
                    value_head_layers.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()
            current_depth = depth

        value_head_layers.append(nn.AdaptiveAvgPool3d(1))
        value_head_layers.append(nn.Flatten())
        value_head_layers.append(nn.Tanh())

        self.value_head = nn.Sequential(*value_head_layers)



    def forward(self, x):
        out = self.value_head(x)
        return out
    
    
# ------------------------------ #
 
class Dense_ValueHead(nn.Module):

    def __init__(self, width, conv_channels=32, batch_norm=False):
        super().__init__()        
        
        layers = []
        
        layers.append(hexagdly.Conv2d(in_channels=width, out_channels=conv_channels, kernel_size=1, stride=1, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=conv_channels))
        layers.append(nn.Flatten())
        layers.append(nn.ReLU())
        layers.append(nn.LazyLinear(256, bias=False))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_features=256, out_features=1, bias=False))
        layers.append(nn.Tanh())
        

        self.value_head = nn.Sequential(*layers)



    def forward(self, x):
        out = self.value_head(x)
        return out
    

##################################################################################################

class Conv_PolicyHead(nn.Module):

    def __init__(self, width, policy_channels, batch_norm=False):
        super().__init__()
        
        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # Filter reduction before last layer
        # number of filters should be close to the dim of the output but not smaller
        
        layers = []
        
        layers.append(hexagdly.Conv2d(in_channels=width, out_channels=policy_filters, kernel_size=1, stride=1, bias=False))
        if batch_norm:
            layers.append(nn.BatchNorm2d(num_features=policy_filters))
        layers.append(nn.ReLU())
        layers.append(hexagdly.Conv2d(in_channels=policy_filters, out_channels=policy_channels, kernel_size=1, stride=1, bias=False))

        self.policy_head = nn.Sequential(*layers)



    def forward(self, x):
        out = self.policy_head(x)
        return out
    
