import sys
from .Unit import Unit


class Tank(Unit):
    mov_points=4
    atack=2
    defense=2

    image_path = "SCS/Images/tank.png"

    def __init__(self, player, x, y, image_path=""):
        super().__init__(player, x, y)

        self.image_path = "SCS/Images/p" + str(player) + "_tank.png"
        if image_path != "":
            self.image_path = image_path
    
        
    def unit_type(self):
        return 2
    
    def unit_name(self):
        return "tank"    
    