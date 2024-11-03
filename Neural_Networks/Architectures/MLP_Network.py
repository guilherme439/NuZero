import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn



class MLP_Network(nn.Module):

    def __init__(self, out_features, hidden_layers=4, neurons_per_layer=64):

        super(MLP_Network, self).__init__()
        self.recurrent=False

        # General Module
        general_module_layers = [nn.Flatten(), nn.LazyLinear(64), nn.SiLU()] # First layer and activation
        for layer in range(hidden_layers):
            general_module_layers.append(nn.Linear(in_features=neurons_per_layer, out_features=neurons_per_layer))
            general_module_layers.append(nn.SiLU())

        self.general_module = nn.Sequential(*general_module_layers)

        

        # Policy Head
        hidden_policy_layers = 3
        policy_head_layers = []
        
        delta = out_features - neurons_per_layer
        step = delta / hidden_policy_layers
        previous_layer_neurons = neurons_per_layer
        for layer in range(hidden_policy_layers):
            current_layer_neurons = previous_layer_neurons + step
            policy_head_layers.append(nn.Linear(in_features=int(previous_layer_neurons), out_features=int(current_layer_neurons)))
            policy_head_layers.append(nn.ReLU())
            previous_layer_neurons = current_layer_neurons
        

        self.policy_head = nn.Sequential(*policy_head_layers)



        # Value Head
        hidden_value_layers = 3
        value_head_layers = []
        
        delta = 1 - neurons_per_layer
        step = delta / hidden_value_layers
        previous_layer_neurons = neurons_per_layer
        for layer in range(hidden_value_layers):
            current_layer_neurons = previous_layer_neurons + step
            value_head_layers.append(nn.Linear(in_features=int(previous_layer_neurons), out_features=int(current_layer_neurons)))
            value_head_layers.append(nn.Tanh())
            previous_layer_neurons = current_layer_neurons
        

        self.value_head = nn.Sequential(*value_head_layers)
        


    def forward(self, x):

        x = self.general_module(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    



    