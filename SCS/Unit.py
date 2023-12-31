import sys
import math
import numpy as np
import sys

'''
from enum import Enum
class Unit_Status(Enum):
    PLACED = 0
    MOVED = 1
    ATTACKED = 2

    def index(self):
        return self.value
'''

class Unit(object):
    player = -1

    mov_allowance = 0
    attack = 0
    defense = 0

    image_path = "SCS/Images/default_unit.webp"


    def __init__(self, name, attack, defense, mov_allowance, player, arraival_locations, image_path):
        self.name = name
        self.attack = attack
        self.defense = defense
        self.mov_allowance = mov_allowance
        self.player=player
        self.arraival_locations = arraival_locations

        if image_path != "":
            self.image_path = image_path
        
        self.mov_points = mov_allowance
        self.status = 0
    
    def get_name(self):
        return self.name
    
    def get_image_path(self):
        return self.image_path
    
    def get_arraival_locations(self):
        return self.arraival_locations
    
    def set_arraival_locations(self, locations_list):
        self.arraival_locations = locations_list
    
    def set_status(self, new_status):
        self.status = new_status
    
    def reset_mov(self):
        self.mov_points = self.mov_allowance
        
    def move_to(self, position, cost):
        self.mov_points -= cost
        self.position = position
    
    def edit_defense(self, new_defense):
        self.defense = new_defense

    def __str__(self):
        string = "P" + str(self.player) + " " + self.name
        if hasattr(self, 'position'):
            string += " at " + "(" + str(self.position[0] + 1) + "," + str(self.position[1] + 1) + ")"
        else:
            string += " not placed yet"
        return string
    
    def __repr__(self):
        return self.__str__()




        
    
        
      