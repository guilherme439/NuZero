import ray
import torch
import time

import numpy as np

from Node import Node
from Explorer import Explorer

from Utils.Caches.DictCache import DictCache
from Utils.Caches.KeylessCache import KeylessCache

from functools import reduce



@ray.remote(scheduling_strategy="SPREAD")
class Gamer():  

    def __init__(self, buffer, shared_storage, game_class, game_args, search_config, recurrent_iterations, cache_choice):

        self.buffer = buffer
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.game_args = game_args

        self.search_config = search_config
        self.recurrent_iterations = recurrent_iterations
        self.cache_choice = cache_choice
        
        self.explorer = Explorer(search_config, True)

        self.time_to_stop = False
        


    def play_game(self):
        future_network = self.shared_storage.get.remote() # ask for a copy of the latest network

        stats = \
        {
        "number_of_moves" : 0,
        "average_children" : 0,
        "average_tree_size" : 0,
        "final_tree_size" : 0,
        "average_bias_value" : 0,
        "final_bias_value" : 0,
        }

        game = self.game_class(*self.game_args)
        keep_subtree = self.search_config.simulation["keep_subtree"]

        if self.cache_choice == "dict":
            self.cache = DictCache()
        elif self.cache_choice == "keyless":
            self.cache = KeylessCache(4096)
        elif self.cache_choice == "disabled":
            self.cache = None
        else:
            print("\nbad cache_choice")
            exit()

        root_node = Node(0)

        network_copy = ray.get(future_network, timeout=200)
        network_copy.check_devices() # Switch to gpu if available

        while not game.is_terminal():
            state = game.generate_state_image()
            game.store_state(state)
            #game.store_player(game.get_current_player())
            
            action_i, chosen_child, root_bias = self.explorer.run_mcts(game, network_copy, root_node, self.recurrent_iterations, self.cache)
            tree_size = root_node.get_visit_count()
            node_children = root_node.num_children()


            action_coords = game.get_action_coords(action_i)
            game.step_function(action_coords)

            game.store_search_statistics(root_node)
            if keep_subtree:
                root_node = chosen_child

            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            stats["average_bias_value"] += root_bias
            stats["final_bias_value"] = root_bias
            
            
        stats["number_of_moves"] = game.length
        stats["average_children"] /= game.length
        stats["average_tree_size"] /= game.length
        stats["average_bias_value"] /= game.length


        ray.get(self.buffer.save_game.remote(game)) # each actor waits for the game to be saved before returning
        return stats
    
    def play_forever(self):
        while not self.time_to_stop:
            self.play_game()

    def stop(self):
        self.time_to_stop = True
    

