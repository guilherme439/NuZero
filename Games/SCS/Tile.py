import math
import numpy as np
import sys

from .Terrain import Terrain

class Tile(object):
    

    def __init__(self, position, terrain=None):
        self.victory = 0            # Integer identifing if the tile is a victory point for either player
        self.terrain = terrain      # Terrain present on the tile
        self.units = []             # List of units in the tile
        self.position = position    # Integer pair representing tile position
        self.player = 0             # Integer identifing the player who currently owns the tile

    def get_terrain(self):
        return self.terrain
    
    def stacking_number(self):
        return len(self.units)
    
    # Stacking level corresponds to index in the units array
    def get_stacking_level(self, unit):
        return self.units.index(unit)
    
    def get_unit_by_level(self, stacking_lvl):
        return self.units[stacking_lvl]
    
    def set_player(self, player):
        self.player = player

    def place_unit(self, unit):
        self.player = unit.player
        self.units.append(unit)

    def remove_unit(self, unit):
        if self.stacking_number() == 1:
            self.player = 0
        self.units.remove(unit)

    def reset(self):
        self.units.clear()
        self.player = 0
    
    def __eq__(self, other): 
        if not isinstance(other, Tile):
            return False

        print("\n\nTile eq not implemented!!\n\n")
        return (self.terrain == other.terrain and self.victory == other.victory and
                self.units == other.units and self.position == other.position)