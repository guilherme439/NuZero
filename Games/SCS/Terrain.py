import math
import numpy as np
import sys

class Terrain():

    name = ""

    # Combat modifiers
    attack_modifier = 1      # Affects who atacks FROM the tile
    defense_modifier = 1    # Affects who defends FROM the tile   

    # Movement costs
    cost = 0

    image_path = "SCS/Images/dirt.jpg"
    
    def __init__(self, attack_modifier, defense_modifier, cost, name="", image_path=""):
        
        self.name=name
        self.attack_modifier=attack_modifier
        self.defense_modifier=defense_modifier
        self.cost=cost

        if image_path != "":
            self.image_path = image_path


    def get_name(self):
        return self.name
    
    def get_image_path(self):
        return self.image_path
    


    
