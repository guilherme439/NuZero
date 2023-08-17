import sys
import math
import numpy as np
import sys


class Unit(object):
    player = -1

    mov_allowance = 0
    attack = 0
    defense = 0

    image_path = "SCS/Images/default_unit.webp"


    def __init__(self, name, attack, defense, mov_allowance, player, image_path):
        self.name = name
        self.attack = attack
        self.defense = defense
        self.mov_allowance = mov_allowance

        self.player=player

        if image_path != "":
            self.image_path = image_path
        
        self.mov_points = self.mov_allowance
    
    def unit_name(self):
        return self.name
    
    def get_image_path(self):
        return self.image_path
    
    def reset_mov(self):
        self.mov_points = self.mov_allowance
        
    def move_to(self, row, col, cost):
        self.mov_points -= cost
        self.row=row
        self.col=col
    
    # Deal damage function
    def edit_defense(self, new_defense):
        self.defense = new_defense

    def __str__(self):
        string = self.name
        if hasattr(self, 'row') and hasattr(self, 'col'):
            string += self.name + " at " + "(" + str(self.row) + ", " + str(self.col) + ")"
        else:
            string += " not placed yet"
        return string
    
    def __repr__(self):
        string = self.name
        if hasattr(self, 'row') and hasattr(self, 'col'):
            string += self.name + " at " + "(" + str(self.row) + ", " + str(self.col) + ")"
        else:
            string += " not placed yet"
        return string




        
    
        
      