import time
import os
import math
import numpy as np
import torch
from torch import nn



class MLP_Network(nn.Module):

    def __init__(self, game):

        super(MLP_Network, self).__init__()
    
        self.action_space_shape = game.get_action_space_shape()
        total_action_planes = list(self.action_space_shape)[0]
        self.input_shape = game.state_shape()
        n_channels = self.input_shape[0]


        # General Module
        self.general_module = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64),
        )

        

        # Policy Head
        size_of_action_space = self.action_space_shape[0] * self.action_space_shape[1] * self.action_space_shape[2]

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.SiLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.SiLU(),
            nn.Linear(in_features=16, out_features=size_of_action_space),
            nn.Softmax(dim=1),
            nn.Unflatten(1, self.action_space_shape)
        )
        


        # Value Head
        self.value_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.Hardtanh(),
            nn.Linear(in_features=32, out_features=16),
            nn.Hardtanh(),
            nn.Linear(in_features=16, out_features=1),
            nn.Tanh()
        )
        


    def forward(self, x):

        x = self.general_module(x)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
    



    