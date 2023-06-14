import sys
import math
import numpy as np
import sys


class Unit(object):
    player = -1
    tile_x = -1
    tile_y = -1

    mov_points = 0
    atack = 0
    defense = 0

    image_path = "SCS/Images/default_unit.webp"

    def __init__(self, player, x, y):
        self.player=player
        self.tile_x=x 
        self.tile_y=y

    def unit_type(self):
        return 0
    
    def unit_name(self):
        return "generic_unit"
    
    def get_image_path(self):
        return self.image_path
    
    def move_to(self, x, y):
        self.tile_x=x
        self.tile_y=y
    
    # Deal damage function
    def edit_defense(self, new_defense):
        self.defense = new_defense