
import numpy as np
from Testing.Agents.Agent import Agent

class RandomAgent(Agent):
    ''' Chooses actions at random (within legal actions)'''

    def __init__(self):
        return

    def choose_action(self, game):
        #possible_actions = game.infos[0]["action_mask"]       
        possible_actions = game.possible_actions().flatten()    # 1.get possible_actions
        #print(game.agent_selection)
        total = sum(possible_actions)
        probs = possible_actions/total          # 2.normalize them 
        print(f"\nTotal valid actions: {total}")
        action_i = np.random.choice(game.num_actions, p=probs)  # 3.select one at random
        return game.get_action_coords(action_i)                 # 4.return it as a coordenate in action space
    
    def name(self):
        return "Random Agent"


          
   