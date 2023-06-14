import math
import numpy as np
import sys

class Terrain():

    Name = ""

    # Combat modifiers
    Atack_modifier = 1
    Defense_modifier = 1

    # Movement costs
    Mov_Add_cost = 0    # Terrain Features
    Mov_Mult_cost = 1   # Terrain Type

    image_path = "Images/dirt.jpg"
    
    def __init__(self, Atack_modifier, Defense_modifier, Mov_Add_cost, Mov_Mult_cost, Name="", image_path=""):
        
        self.Name=Name
        self.Atack_modifier=Atack_modifier
        self.Defense_modifier=Defense_modifier
        self.Mov_Add_cost=Mov_Add_cost
        self.Mov_Mult_cost=Mov_Mult_cost

        if image_path != "":
            self.image_path = image_path


    def get_name(self):
        return self.Name
    
    def get_image_path(self):
        return self.image_path
    


    
