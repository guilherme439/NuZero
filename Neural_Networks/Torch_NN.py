import time
import os
import math
import numpy as np


from collections import Counter

import torch
from torch import nn


class Torch_NN():

    def __init__(self, game, model):
        
        self.model = model
        self.check_devices()

        self.model = model.to(self.device)
        if not hasattr(self.model, "recurrent"):
            print("You need to add a \"recurrent\" bollean atribute to the model,\n \
                   Specifing if the model is or not recurrent.")
            exit()
        elif not isinstance(self.model.recurrent, bool):
            print("\"model.recurrent\" must be a bollean atribute specifing if the model is or not recurrent.")
            exit()

        self.game = game
        

    def get_model(self):
        return self.model
    
    def model_to_cpu(self):
        self.model = self.model.to('cpu')

    def model_to_device(self):
        self.model = self.model.to(self.device)

    def cuda_is_available():
        return torch.cuda.is_available()

    def check_devices(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def inference(self, state, training, iters_to_do=2, interim_thought=None):

        if not training:
            self.model.eval()

        if not self.model.recurrent:
            if not training:
                with torch.no_grad():
                    p,v = self.model(state.to(self.device))
            else:
                p,v = self.model(state.to(self.device))
        else:
            if not training:
                with torch.no_grad():
                    (p,v), _ = self.model(state.to(self.device), iters_to_do)
            else:
                return self.model(state.to(self.device), iters_to_do, interim_thought)

        return p,v
    
    

    

    



    