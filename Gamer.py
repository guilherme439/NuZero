import ray
import torch

import numpy as np

from Node import Node
from Explorer import Explorer



@ray.remote(scheduling_strategy="SPREAD")
class Gamer():  

    def __init__(self, buffer, shared_storage, game_class, game_args, search_config, num_iterations, state_cache):

        self.buffer = buffer
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.game_args = game_args

        self.search_config = search_config
        self.state_cache = state_cache

        if self.state_cache == "per_actor":
            self.state_table = {}
        else:
            self.state_table = None
        
        self.explorer = Explorer(search_config, True, num_iterations)
        

    def play_game(self): 
        torch.multiprocessing.set_sharing_strategy('file_system')
        
        future_network = self.shared_storage.get_latest_network.remote() # ask for latest network

        if self.state_cache == "per_game":
            self.state_table = {}

        keep_sub_tree = self.search_config.simulation["keep_sub_tree"]
    
        subtree_root = Node(0)
        game = self.game_class(*self.game_args)
        network = ray.get(future_network, timeout=60)
        while not game.is_terminal():
            state = game.generate_state_image()
            game.store_state(state)
            
            if not keep_sub_tree:
                subtree_root = Node(0)

            action_i, chosen_child = self.explorer.run_mcts(network, game, subtree_root, self.state_table)
            action_coords = np.unravel_index(action_i, game.get_action_space_shape())

            game.step_function(action_coords)

            game.store_search_statistics(subtree_root)
            if keep_sub_tree:
                subtree_root = chosen_child
       

        ray.get(self.buffer.save_game.remote(game)) # each actor waits for the game to be saved before returning
        return

