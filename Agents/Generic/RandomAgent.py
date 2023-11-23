from SCS.SCS_Game import SCS_Game

import numpy as np

    

class RandomAgent():
    ''' Chooses actions at random (within legal actions)'''

    def __init__(self):
        return

    def choose_action(self, game):       
        possible_actions = game.possible_actions().flatten()    # 1.get possible_actions
        probs = possible_actions/sum(possible_actions)          # 2.normalize them 
        action_i = np.random.choice(game.num_actions, p=probs)  # 3.select one at random
        return game.get_action_coords(action_i)                 # 4.return it as a coordenate in action space


          
   