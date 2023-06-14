import math
import numpy as np
import sys

from .Terrain import Terrain

class Tile(object):
    
    victory=0
    terrain=None
    unit=None

    def __init__(self, terrain=None):
        self.vicotry = 0
        self.terrain = terrain

    def get_terrain(self):
        return self.terrain

    def place_unit(self,unit):
        self.unit=unit

    
    def __eq__(self, other): 
        if not isinstance(other, Tile):
            return False

        print("\n\nNot implemented!!\n\n")
        return self.terrain == other.terrain and self.victory == other.vicotry and self.unit == other.unit