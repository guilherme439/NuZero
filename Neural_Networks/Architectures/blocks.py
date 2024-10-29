import math

import torch
from torch import nn
import torch.nn.functional as F

import hexagdly

from .depthwise_conv import depthwise_conv


class BasicBlock(nn.Module):

    def __init__(self, channels, batch_norm=False, hex=True):
        super().__init__()
        
        before_shortcut_layers = []

        if hex: 
            before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        else:
            before_shortcut_layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 3, padding='same', bias=False))
        if batch_norm:
            before_shortcut_layers.append(nn.BatchNorm2d(num_features=channels))
        before_shortcut_layers.append(nn.ReLU())

        if hex:
            before_shortcut_layers.append(hexagdly.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 1, bias=False))
        else:
            before_shortcut_layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size = 3, padding='same', bias=False))
        
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
       followed by a global average pooling layer'''

    def __init__(self, width, num_reduce_layers=4, activation="tanh", batch_norm=False, hex=True):
        super().__init__()

        value_head_layers = []
        final_layer_filters = 1

        delta = final_layer_filters - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers,0,-1):
            current_layer_filters = previous_layer_filters + step
            if hex:
                conv = hexagdly.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=1, stride=1, bias=False)
            else:
                conv = nn.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=3, stride=1, padding='same', bias=False)
            value_head_layers.append(conv)

            if layer != 1:
                if batch_norm:
                    value_head_layers.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))

                if activation == "tanh":
                    value_head_layers.append(nn.Tanh())
                elif activation == "relu":
                    value_head_layers.append(nn.ReLU())
                else:
                    print("Unknown activation.")
                    exit()

            previous_layer_filters = current_layer_filters

        value_head_layers.append(nn.AdaptiveAvgPool3d(1))
        value_head_layers.append(nn.Flatten())
        value_head_layers.append(nn.Tanh())

        self.layers = nn.Sequential(*value_head_layers)



    def forward(self, x):
        out = self.layers(x)
        return out
    

# ------------------------------ #
 
class Dense_ValueHead(nn.Module):

    def __init__(self, width, dense_layer_neurons=256, conv_layer_channels=32, batch_norm=False, hex=True):
        super().__init__()        
        
        layer_list = []
        
        if hex:
            layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=conv_layer_channels, kernel_size=1, stride=1, bias=False))
        else:
            layer_list.append(nn.Conv2d(in_channels=width, out_channels=conv_layer_channels, kernel_size=3, stride=1, padding='same', bias=False))

        if batch_norm:
            layer_list.append(nn.BatchNorm2d(num_features=conv_layer_channels))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.ReLU())
        layer_list.append(nn.LazyLinear(dense_layer_neurons, bias=False))
        layer_list.append(nn.ReLU())
        layer_list.append(nn.Linear(in_features=dense_layer_neurons, out_features=1, bias=False))
        layer_list.append(nn.Tanh())
        

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    

##################################################################################################

class Reduce_PolicyHead(nn.Module):
    '''Several conv layers that progressively reduce the amount of filters,
       until the policy's number of channels is reached'''

    def __init__(self, width, policy_channels, num_reduce_layers=2, batch_norm=False, hex=True):
        super().__init__()

        '''
        final_policy_filters = int(math.pow(2, math.ceil(math.log(policy_channels, 2)))) # Filter reduction before last layer
        # number of filters should be close to the dim of the output but not smaller
        '''

        layer_list = []

        delta = policy_channels - width
        step = delta / num_reduce_layers
        previous_layer_filters = width

        for layer in range(num_reduce_layers, 0, -1):
            current_layer_filters = previous_layer_filters + step
            if hex:
                layer_list.append(hexagdly.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=1, stride=1, bias=False))
            else:
                layer_list.append(nn.Conv2d(in_channels=int(previous_layer_filters), out_channels=int(current_layer_filters), kernel_size=3, stride=1, padding='same', bias=False))

            if layer != 1:
                if batch_norm:
                    layer_list.append(nn.BatchNorm2d(num_features=int(current_layer_filters)))

                layer_list.append(nn.ReLU())

            previous_layer_filters = current_layer_filters


        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    

###################################################################################################################################################
###################################################################################################################################################
#                                                                  DISCONTINUED                                                                   #
###################################################################################################################################################
###################################################################################################################################################


class Depth_ValueHead(nn.Module):

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
        super().__init__()

        layer_list = []
        num_depth_layers = 4

        for l in range(num_depth_layers):
            if hex:
                layer_list.append(depthwise_conv(in_channels=width, out_channels=width, kernel_size=1, stride=1, bias=False))
            else:
                layer_list.append(nn.Conv2d(in_channels=width, out_channels=width, groups=width, kernel_size=3, stride=1, bias=False))
            if batch_norm:
                layer_list.append(nn.BatchNorm2d(num_features=width))

            if activation == "tanh":
                layer_list.append(nn.Tanh())
            elif activation == "relu":
                layer_list.append(nn.ReLU())
            else:
                print("Unknown activation.")
                exit()

        if hex:
            layer_list.append(hexagdly.Conv2d(in_channels=width, out_channels=1, kernel_size=1, stride=1, bias=False))
        else:
            layer_list.append(nn.Conv2d(in_channels=width, out_channels=1, kernel_size=3, stride=1, bias=False))

        layer_list.append(nn.AdaptiveAvgPool3d(1))
        layer_list.append(nn.Flatten())
        layer_list.append(nn.Tanh())

        self.layers = nn.Sequential(*layer_list)



    def forward(self, x):
        out = self.layers(x)
        return out
    
# ------------------------------ #

class Combined_ValueHead(nn.Module):

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
        super().__init__()
        
        layer_list = []
        conv_filters = [256, 64, 8 , 1]
        
        current_filters = width
        for filters in conv_filters:
            if hex:
                layer_list.append(depthwise_conv(in_channels=current_filters, out_channels=current_filters, kernel_size=1, stride=1, bias=False))
            else:
                layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=current_filters, groups=current_filters, kernel_size=3, stride=1, bias=False))
            if batch_norm:
                layer_list.append(nn.BatchNorm2d(num_features=current_filters))

            if activation == "tanh":
                layer_list.append(nn.Tanh())
            elif activation == "relu":
                layer_list.append(nn.ReLU())
            else:
                print("Unknown activation.")
                exit()
            if hex:
                layer_list.append(hexagdly.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False))
            else:
                layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=3, stride=1, bias=False))

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

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            if hex:
                layer_list.append(depthwise_conv(in_channels=current_filters, out_channels=current_filters, kernel_size=1, stride=1, bias=False)) #depthwise
            else:
                layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=current_filters, groups=current_filters, kernel_size=3, stride=1, bias=False)) #depthwise

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

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # pointwise
            if hex:
                layer_list.append(depthwise_conv(in_channels=filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) #depthwise
            else:
                layer_list.append(nn.Conv2d(in_channels=filters, out_channels=filters, groups=filters, kernel_size=3, stride=1, bias=False)) #depthwise


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

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
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

    def __init__(self, width, activation="relu", batch_norm=False, hex=True):
        super().__init__()
        
        conv_filters = [256, 64 , 8 , 1]

        layer_list = []
        current_filters = width
        for filters in conv_filters:
            layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=current_filters, kernel_size=1, groups=current_filters, stride=1, bias=False)) # depth pointwise
            if hex:
                layer_list.append(hexagdly.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=1, stride=1, bias=False)) # normal conv
            else:
                layer_list.append(nn.Conv2d(in_channels=current_filters, out_channels=filters, kernel_size=3, stride=1, bias=False)) # normal conv

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
    
    
    
