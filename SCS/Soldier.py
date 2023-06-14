import sys
from .Unit import Unit

class Soldier(Unit):
    mov_points=2
    atack=1
    defense=1

    image_path = "SCS/Images/soldier.png"

    def __init__(self, player, x, y, image_path=""):
        super().__init__(player, x, y)

        self.image_path = "SCS/Images/p" + str(player) + "_soldier.png"
        if image_path != "":
            self.image_path = image_path
    
    
    def unit_type(self):
        return 1

    def unit_name(self):
        return "soldier"
    