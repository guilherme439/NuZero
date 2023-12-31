from SCS.SCS_Game import SCS_Game

import numpy as np
from scipy.special import softmax

from Agents.Agent import Agent
    

class PolicyAgent(Agent):
    ''' Chooses actions acording to a neural network's policy'''

    def __init__(self, network, recurrent_iterations=2):
        self.network = network
        self.recurrent_iterations = recurrent_iterations
        return

    def choose_action(self, game):
        
        state = game.generate_state_image()
        policy_logits, value_pred = self.network.inference(state, False, self.recurrent_iterations)
        probs = softmax(policy_logits).flatten()

        raw_action = np.argmax(probs)
        valid_actions_mask = game.possible_actions().flatten()
        n_valids = sum(valid_actions_mask)
        if not valid_actions_mask[raw_action]:
            # Check if the network returned a possible action,
            # if it didn't, do the necessary workarounds
            
            probs = probs * valid_actions_mask
            total = np.sum(probs)

            if (total != 0): 
                probs /= total
                chance_action = np.random.choice(game.num_actions, p=probs)

                max_action = np.argmax(probs)
                action_i = max_action

            else:
                # happens if the network gave 0 probablity to all valid actions and high probability to invalid actions
                # There was a problem in training... using random action instead
                probs = probs + valid_actions_mask
                probs /= n_valids
                action_i = np.random.choice(game.num_actions, p=probs)
        
        else:
            action_i = raw_action

        return game.get_action_coords(action_i)
    
    def name(self):
        return "Policy Agent"

          
   