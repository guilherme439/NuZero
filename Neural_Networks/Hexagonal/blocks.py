
import math

import torch
from torch import nn
import torch.nn.functional as F

import hexagdly

from .depthwise_conv import depthwise_conv


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
    '''Several conv layers that progressively reduce the amount of filters,
       followed by an average pooling layer'''

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        conv_filters = [256, 64, 8 , 1]

        value_head_layers = []
        current_depth = width
        for depth in conv_filters:
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

        self.layers = nn.Sequential(*value_head_layers)



    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class Depth_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()

        layer_list = []
        num_depth_layers = 4

        for l in range(num_depth_layers):
            layer_list.append(depthwise_conv(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=False))
            if batch_norm:
                layer_list.append(nn.BatchNorm2d(num_features=width))

            if activation == "tanh":
                layer_list.append(nn.Tanh())
            elif activation == "relu":
                layer_list.append(nn.ReLU())
            else:
                print("Unknown activation.")
                exit()

        layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=1, kernel_size=1, stride=1, bias=False))

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class Combined_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        layer_list = []
        conv_filters = [256, 64, 8 , 1]
        
        current_filters = width
        for filters in conv_filters:
            layer_list.append(depthwise_conv(in_channels=current_filters, out_channels=current_filters, kernel_size=1, stride=1, bias=False))
            if batch_norm:
                layer_list.append(nn.BatchNorm2d(num_features=current_filters))

            if activation == "tanh":
                layer_list.append(nn.Tanh())
            elif activation == "relu":
                layer_list.append(nn.ReLU())
            else:
                print("Unknown activation.")
                exit()

            layer_list.append(hexagdly.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False))
            if filters != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=filters))

                if activation == "tanh":
                    layer_list.append(nn.Tanh())
                elif activation == "relu":
                    layer_list.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()
            
            current_filters = filters


        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)

    


    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class Separable_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(depthwise_conv(in_channels=current_filters, out_channels=current_filters, kernel_size=1, stride=1, bias=False)) #depthwise
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # pointwise
            current_filters=filters

            if filters != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=filters))

                if activation == "tanh":
                    layer_list.append(nn.Tanh())
                elif activation == "relu":
                    layer_list.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        out = self.layers(x)
        return out

# ------------------------------ #

class Reverse_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # pointwise
            layer_list.append(depthwise_conv(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) #depthwise
            current_filters=filters

            if filters != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=filters))

                if activation == "tanh":
                    layer_list.append(nn.Tanh())
                elif activation == "relu":
                    layer_list.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class RawSeparable_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=current_filters, kernel_size=3, groups=current_filters, stride=1, bias=False)) #depthwise
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # pointwise
            current_filters=filters

            if filters != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=filters))

                if activation == "tanh":
                    layer_list.append(nn.Tanh())
                elif activation == "relu":
                    layer_list.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class Strange_ValueHead(nn.Module):

    def __init__(self, width, activation, batch_norm=False):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=current_filters, kernel_size=1, groups=current_filters, stride=1, bias=False)) # depth pointwise
            layer_list.append(hexagdly.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # normal hex conv
            current_filters=filters

            if filters != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=filters))

                if activation == "tanh":
                    layer_list.append(nn.Tanh())
                elif activation == "relu":
                    layer_list.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)


    def forward(self, x):
        out = self.layers(x)
        return out
    
    
# ------------------------------ #
 
class Dense_ValueHead(nn.Module):

    def __init__(self, width, conv_channels=32, batch_norm=False):
        super().__init__()        
        
        layer_list = []
        
        layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=conv_channels, kernel_size=1, stride=1, bias=False))
        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=conv_channels))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.ReLU())
        layer_list.append(nn.LazyLinear(256, bias=False))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(in_features=256, out_features=1, bias=False))
        layer_list.append(nn.Tanh())
        

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    

##################################################################################################

class Conv_PolicyHead(nn.Module):

    def __init__(self, width, policy_channels, batch_norm=False):
        super().__init__()
        

        first_reduction = 128

        policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # Filter reduction before last layer
        # number of filters should be close to the dim of the output but not smaller
        
        layer_list = []

        layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=first_reduction, kernel_size=1, stride=1, bias=False))
        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=first_reduction))
        layer_list.append(nn.ReLU())
        layer_list.append(hexagdly.Conv2d(in_channels=first_reduction, out_channels=policy_filters, kernel_size=1, stride=1, bias=False))
        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=policy_filters))
        layer_list.append(nn.ReLU())
        layer_list.append(hexagdly.Conv2d(in_channels=policy_filters, out_channels=policy_channels, kernel_size=1, stride=1, bias=False))

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    
