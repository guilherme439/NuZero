from SCS.SCS_Game import SCS_Game

import numpy as np
from Node import Node
from Explorer import Explorer
from Agents.Agent import Agent

from Utils.other_utils import *


class MctsAgent(Agent):
    ''' Chooses the action most visited using AlphaZero's MCTS'''

    def __init__(self, search_config, network, recurrent_iterations=2, cache=None):
        self.explorer = Explorer(search_config, False)
        self.keep_subtree = search_config["Simulation"]["keep_subtree"]
        self.root_node = Node(0)

        self.network = network
        self.recurrent_iterations = recurrent_iterations
        self.cache = cache

        return

    def choose_action(self, game):
        action_i, chosen_child, root_bias = self.explorer.run_mcts(game, self.network, self.root_node, self.recurrent_iterations, self.cache)
        if self.keep_subtree:
            self.root_node = chosen_child

        return game.get_action_coords(action_i)
    
    def update_subtree(self, game, action_i):
        # In order to update the subtree we need to run the mcts once again and then select the correct node acording to the chosen action
        _, _, _ = self.explorer.run_mcts(game, self.network, self.root_node, self.recurrent_iterations, self.cache)
        self.root_node = self.root_node.get_child(action_i)
        return
    
    
    def new_game(self, *args, cache=None):
        self.root_node = Node(0)
        if cache is not None:
            self.cache = cache

    def get_cache(self):
        return self.cache
    
    def set_cache(self, cache):
        self.cache = cache
    
    def name(self):
        return "MCTS Agent"
    


          
    