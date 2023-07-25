import time
import os
import math
import numpy as np


from collections import Counter

import torch
from torch import nn


class Torch_NN():

    def __init__(self, game, model, recurrent):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"\n--------------\nUsing {self.device} device\n--------------\n")

        self.model = model.to(self.device)
        self.recurrent = recurrent
        self.game = game
        

    def get_model(self):
        return self.model
    
    def inference(self, state, training, iters_to_do=2):

        if not training:
            self.model.eval()

        if not self.recurrent:
            if not training:
                with torch.no_grad():
                    p,v = self.model(state.to(self.device))
            else:
                p,v = self.model(state.to(self.device))
        else:
            if not training:
                with torch.no_grad():
                    p,v = self.model(state.to(self.device), iters_to_do)
            else:
                p,v = self.model(state.to(self.device), iters_to_do)

        p = torch.unflatten(p, 1, self.game.get_action_space_shape())
        return p,v
    
    

    

    



    