import sys

import ray
import torch

import math
import numpy as np

from Node import Node

from Replay_Buffer import Replay_Buffer


@ray.remote(scheduling_strategy="SPREAD")
class Gamer():  

    def __init__(self, buffer, shared_storage, config, game_class, game_args):

        self.config = config
        self.buffer = buffer
        self.shared_storage = shared_storage
        self.game_class = game_class
        self.game_args = game_args
        self.network = None
        

    def play_game(self, show): 
        
        future_network = self.shared_storage.get_latest_network.remote() # ask for latest network

        torch.multiprocessing.set_sharing_strategy('file_system')

        state_table = {}
        game = self.game_class(*self.game_args)
        subtree_root = Node(0)

        self.network = ray.get(future_network, timeout=60)
        while not game.is_terminal():
            state = game.generate_state_image()
            game.store_state(state)
            game.store_player(game.current_player)
            
            if not self.config.keep_sub_tree:
                subtree_root = Node(0)

            action_i, chosen_child = self.run_mcts(game, subtree_root, state_table)
            action_coords = np.unravel_index(action_i, game.get_action_space_shape())

            game.step_function(action_coords)

            game.store_search_statistics(subtree_root)
            if self.config.keep_sub_tree:
                subtree_root = chosen_child
       

        ray.get(self.buffer.save_game.remote(game)) # each actor waits for the game to be saved before returning
        return

    def run_mcts(self, game, subtree_root, state_table):

        search_start = subtree_root
        self.add_exploration_noise(search_start)
        
        num_searches = self.config.mcts_simulations
        for i in range(num_searches):
            node = search_start
            scratch_game = game.clone()
            search_path = [node]
            
            while node.expanded():
                action_i, node = self.select_child(node)
                action_coords = np.unravel_index(action_i, scratch_game.get_action_space_shape())
                scratch_game.step_function(action_coords)
                search_path.append(node)
            
            if node.is_terminal():
                value = node.value()
            else:
                value = self.evaluate(node, scratch_game, state_table)
            
            self.backpropagate(search_path, value)

        
        action = self.select_action(game, search_start)
        return action, search_start.children[action]
    
    def select_action(self, game, node):
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]

        epsilon = np.random.random()
        if epsilon < self.config.epsilon_random_exploration:
            valid_actions_mask = game.possible_actions().flatten()
            n_valids = np.sum(valid_actions_mask)
            probs = valid_actions_mask
            probs = probs/n_valids
            action_i = np.random.choice(game.get_num_actions(), p=probs)

        elif game.get_length() < self.config.number_of_softmax_moves:
            action_i = self.softmax_action(visit_counts)

        else:
            action_i = self.max_action(visit_counts)


        return action_i
    
    def select_child(self, node):
        _, action, child = max((self.ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child
    
    def ucb_score(self, parent, child):
        pb_c_base = self.config.pb_c_base
        pb_c_init = self.config.pb_c_init

        bias = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c = (math.sqrt(parent.visit_count) / (child.visit_count + 1)) * bias

        prior_score = pb_c * child.prior


        value_score = child.value()
        if parent.to_play == 2:
            value_score = (-value_score)
        # for player 2 negative values are good

        return prior_score + value_score

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value	

    def evaluate(self, node, game, state_table):
        
        node.to_play = game.get_current_player()
        
        if self.config.use_terminal and game.is_terminal():
            node.terminal_value = game.get_terminal_value()
            value = node.terminal_value
            return value
        

        state = game.generate_state_image()
        if self.config.with_cache:
            
            state_index = tuple(state.numpy().flatten())
            result = state_table.get(state_index)
            
            if result:
                (action_probs, predicted_value) = result
            else:
                action_probs, predicted_value = self.network.inference(state, False, self.config.num_pred_iters)
                state_table[state_index] = (action_probs, predicted_value)
                

        else:
            action_probs, predicted_value = self.network.inference(state, False, self.config.num_pred_iters)


        value = predicted_value
        
        if not game.is_terminal():

            # Expand the node.
            valid_actions_mask = game.possible_actions().flatten()
            action_probs = action_probs.cpu()[0].numpy().flatten()
            
            
            probs = action_probs * valid_actions_mask # Use mask to get only valid moves
            total = np.sum(probs)


            for i in range(game.get_num_actions()):
                if valid_actions_mask[i]:
                    node.children[i] = Node(probs[i]/total)

        else:
            node.terminal_value = game.get_terminal_value()
            

        return value 

    def softmax_action(self, visit_counts):
        counts = []
        actions = []
        for (count, action) in visit_counts:
            counts.append(count)	
            actions.append(action)

        #final_counts = softmax(counts)
        final_counts = counts/np.sum(counts)

        probs = np.asarray(final_counts, dtype=np.float64).astype('float64')
        probs /= np.sum(probs) # re-normalize to improve precison
        return np.random.choice(actions, p=probs)
    
    def max_action(self, visit_counts):
        max_pair = max(visit_counts, key=lambda visit_action_pair: visit_action_pair[0])
        return max_pair[1]

    def add_exploration_noise(self, node):
        actions = node.children.keys()
        noise = np.random.gamma(self.config.dist_alpha, self.config.dist_beta, len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def print_tree(self, root, action_space_shape):
        buffer = []
        print("\nRoot -> ")
        buffer.append((root, None, 0, None, 0))

        identifier = 0
        last_pid = 0
        while len(buffer) > 0:
            identifier +=1
            (node, parent, p_id, action_coords, level) = buffer.pop(0)
            if last_pid != p_id:
                print("\n-\n")
            last_pid = p_id
            value_score = node.value()


            if parent:
                ucbScore = self.ucb_score(parent, node)
                print("Level: " + str(level) + " Parent_id: " + str(p_id) + " Node_id: " + format(identifier, '2') + 
                        " V: " + format(value_score, '.02') + " U: " + format(ucbScore, '.2') + " Visits: " + str(node.visit_count) + " To_play: " + str(node.to_play) +
                        " Terminal: " + str(node.terminal_value) + " NN_Prior: " + format(node.prior, '.02'))
                #if node.terminal_value != None:
                    #print("\n" + self.game.string_action(action_coords) + "\n\n")
            else:
                print("Node_id: " + str(identifier) + " Level: " + str(level) + " V: " + format(value_score, '.02') + " To_play: " + str(node.to_play))
            
            
            for (a, child) in node.children.items():
                if child.to_play!=-1:
                    act = np.unravel_index(a, action_space_shape)
                    buffer.append((child, node, identifier, act, level+1))		
        


        #print(colored("A: " + "(" + str(action_coords[0]+1) + "," + str(action_coords[1]+1) + ")" + " V: " + format(value_score, '.02') + " U: " + format(ucbScore, '.2') + " Visits: " + str(child.visit_count), "white"))
        return