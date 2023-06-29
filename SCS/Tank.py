import sys
from .Unit import Unit


class Tank(Unit):
    mov_allowance=4
    atack=2
    defense=4

    image_path = "SCS/Images/tank.png"

    def __init__(self, player, x, y, image_path=""):
        super().__init__(player, x, y)

        self.image_path = "SCS/Images/p" + str(player) + "_tank.png"
        if image_path != "":
            self.image_path = image_path
        
        self.mov_points = self.mov_allowance
    
        
    def unit_type(self):
        return 2
    
    def unit_name(self):
        return "tank"    
    
    '''
    def __eq__(self, other): 
        if not isinstance(other, Tank):
            return False

        print("\n\nTank eq not implemented!!\n\n")
        return False
    '''