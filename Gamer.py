import ray
import torch
import time

import numpy as np

from Node import Node
from Explorer import Explorer



@ray.remote(scheduling_strategy="SPREAD")
class Gamer():  

    def __init__(self, buffer, shared_storage, game_class, game_args, search_config, recurrent_iterations, state_cache):

        self.buffer = buffer
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.game_args = game_args

        self.search_config = search_config
        self.state_cache = state_cache

        if self.state_cache == "per_actor":
            self.state_dict = {}
        else:
            self.state_dict = None
        
        self.explorer = Explorer(search_config, True, recurrent_iterations)
        

    def play_game(self):
        torch.multiprocessing.set_sharing_strategy('file_system')
        future_network = self.shared_storage.get_latest_network.remote() # ask for latest network

        mcts = 0

        stats = \
        {
        "number_of_moves" : 0,
        "average_children" : 0,
        "final_tree_size" : 0,
        "average_tree_size" : 0,
        "final_bias_value" : 0,
        "average_bias_value" : 0,
        "average_value_score" : 0,
        "average_prior_score" : 0,
        }

        if self.state_cache == "per_game":
            self.state_dict = {}

        keep_sub_tree = self.search_config.simulation["keep_sub_tree"]
        
        subtree_root = Node(0)
        game = self.game_class(*self.game_args)
        network = ray.get(future_network, timeout=120)
        while not game.is_terminal():
            state = game.generate_state_image()
            game.store_state(state)
            
            if not keep_sub_tree:
                subtree_root = Node(0)

            second = time.time()
            
            action_i, chosen_child, search_stats = self.explorer.run_mcts(network, game, subtree_root, self.state_dict)
            tree_size = subtree_root.get_visit_count()
            node_children = subtree_root.num_children()

            third = time.time()
            mcts += (third-second)

            action_coords = np.unravel_index(action_i, game.get_action_space_shape())
            game.step_function(action_coords)

            game.store_search_statistics(subtree_root)
            if keep_sub_tree:
                subtree_root = chosen_child

            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            stats["average_bias_value"] += search_stats["root_bias_value"]
            stats["final_bias_value"] = search_stats["root_bias_value"]
            stats["average_prior_score"] += search_stats["average_prior_score"]
            stats["average_value_score"] += search_stats["average_value_score"]

            
            
        stats["number_of_moves"] = game.length
        stats["average_children"] /= game.length
        stats["average_tree_size"] /= game.length
        stats["average_bias_value"] /= game.length
        stats["average_prior_score"] /= game.length
        stats["average_value_score"] /= game.length

        print()
        print("mcts: " + format(mcts/game.length, '.4'))


        ray.get(self.buffer.save_game.remote(game)) # each actor waits for the game to be saved before returning
        return stats

