import numpy as np
import hexagdly
import torch
import math
import time
import os

from torch import nn



class MLP_Network(nn.Module):

    def __init__(self, out_features):

        super(MLP_Network, self).__init__()
    

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

        self.policy_head = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.SiLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.SiLU(),
            nn.Linear(in_features=16, out_features=out_features),
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
    



    