import math
import numpy as np
import sys

from .Terrain import Terrain

class Tile(object):
    

    def __init__(self, x, y, terrain=None):
        self.victory = 0
        self.terrain = terrain
        self.unit = None
        self.x = x
        self.y = y

    def get_terrain(self):
        return self.terrain

    def place_unit(self,unit):
        self.unit=unit

    def reset(self):
        self.unit = None
    
    def __eq__(self, other): 
        if not isinstance(other, Tile):
            return False

        print("\n\nTile eq not implemented!!\n\n")
        return (self.terrain == other.terrain and self.victory == other.vicotry and
                self.unit == other.unit and self.x == other.x and self.y == other.y)