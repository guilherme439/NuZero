import sys
from .Unit import Unit

class Soldier(Unit):
    mov_allowance=2
    atack=1
    defense=2

    image_path = "SCS/Images/soldier.png"

    def __init__(self, player, x, y, image_path=""):
        super().__init__(player, x, y)

        self.image_path = "SCS/Images/p" + str(player) + "_soldier.png"
        if image_path != "":
            self.image_path = image_path
    
        self.mov_points = self.mov_allowance

    def unit_type(self):
        return 1

    def unit_name(self):
        return "soldier"
    
    '''
    def __eq__(self, other): 
        if not isinstance(other, Soldier):
            return False

        print("\n\nSoldier eq not implemented!!\n\n")
        return False
    '''