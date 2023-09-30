import math
import time
import ray

from scipy.special import softmax

from progress.bar import ChargingBar
from progress.spinner import PieSpinner

import numpy as np

import torch

from Node import Node
from Explorer import Explorer

from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Neural_Networks.Torch_NN import Torch_NN



class Tester():

    def __init__(self, slow=False, print=False, render=False):
        torch.multiprocessing.set_sharing_strategy('file_system')

        self.slow = slow
        self.print = print

        self.render = render
        if render == True:
            self.slow=True
            # Render is only supported for SCS games
            from SCS.SCS_RemoteRenderer import SCS_RemoteRenderer
            from RemoteStorage import RemoteStorage

            self.remote_storage = RemoteStorage.remote()
            self.renderer = SCS_RemoteRenderer.remote(self.remote_storage)

        self.slow_duration = 2

    def set_slow_duration(self, seconds):
        self.slow_duration = seconds
    
# ------------------------------------------------ #
# ----------------- TEST METHODS ----------------- #
# ------------------------------------------------ #

    def Test_AI_with_mcts(self, player_choice, search_config, game, nn1, nn2=None, recurrent_iterations=1, use_state_cache=False, keep_state_history=False):

        # --- Initialization --- #
        stats = \
        {
        "number_of_moves" : 0,
        "average_children" : 0,
        "average_tree_size" : 0,
        "final_tree_size" : 0,
        "average_bias_value" : 0,
        "final_bias_value" : 0,
        }
        keep_sub_tree = search_config.simulation["keep_sub_tree"]
        explorer = Explorer(search_config, False, recurrent_iterations)

        # --- State cache verfication --- #
        if not isinstance(use_state_cache, bool):
            print("\"use_state_cache\" is not a boolean.")
            print("\"use_state_cache\" should be a boolean, representing whether or not to use a state dictionary as cache during this test.")
            print("Exiting...")
            exit()

        if use_state_cache:
            state_dict = {}
        else:
            state_dict = None

        # --- Neural network setup --- #
        # Both players use the first neural network unless a second (nn2) is given
        p1_nn = nn1
        p2_nn = nn1
        if player_choice == "both":
            AI_player = 0
            if nn2:
                p2_nn = nn2

        elif player_choice == "1":
            AI_player = 1
        elif player_choice == "2":
            AI_player = 2
        else:
            print("player_choice should be on these strings: \"1\" | \"2\" | \"both\". Exiting")
            exit()
        
        # --- Printing and rendering preparations --- #
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote(player_unit_images=True)
            time.sleep(3)

        # --- Main test loop --- #
        subtree_root = Node(0)
        while True:
            
            valid_actions_mask = game.possible_actions().flatten()
            n_valids = np.sum(valid_actions_mask)
            if (n_valids == 0):
                print("Zero valid actions!")
                exit()

            if not keep_sub_tree:
                subtree_root = Node(0)

            player = game.current_player
            if player == 1:
                net_to_use = p1_nn
            else:
                net_to_use = p2_nn

            # Use neural network or random choice according to the player #
            if (AI_player == 0) or (player == AI_player):
                action_i, chosen_child, root_bias = explorer.run_mcts(net_to_use, game, subtree_root, state_dict=state_dict)

            else:
                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.get_num_actions(), p=probs)
                if keep_sub_tree:    
                    _, _, root_bias = explorer.run_mcts(net_to_use, game, subtree_root, state_dict=state_dict)
                    chosen_child = subtree_root.children[action_i]

            tree_size = subtree_root.get_visit_count()
            node_children = subtree_root.num_children()

            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)

            if keep_state_history:
                state = game.generate_state_image()
                game.store_state(state)

            # Step function #
            action_coords = np.unravel_index(action_i, game.action_space_shape)
            done = game.step_function(action_coords)

            # Update stats #
            stats["average_children"] += node_children
            stats["average_tree_size"] += tree_size
            stats["final_tree_size"] = tree_size
            if keep_sub_tree:
                stats["average_bias_value"] += root_bias
                stats["final_bias_value"] = root_bias

                subtree_root = chosen_child

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))


            if (done):
                if self.print:
                    print(game.string_representation())
                winner = game.get_winner()
                break


        stats["number_of_moves"] = game.length
        stats["average_children"] /= game.length
        stats["average_tree_size"] /= game.length
        if keep_sub_tree:
            stats["average_bias_value"] /= game.length   


        return winner, stats

    def Test_AI_with_policy(self, player_choice, game, nn1, nn2=None, recurrent_iterations=1, keep_state_history=False):
        
        # --- Printing and rendering preparations --- #
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote(player_unit_images=True)
            time.sleep(3)

        # --- Neural network setup --- #
        # Both players use the first neural network unless a second (nn2) is given
        p1_nn = nn1
        p2_nn = nn1
        if player_choice == "both":
            AI_player = 0
            if nn2:
                p2_nn = nn2

        elif player_choice == "1":
            AI_player = 1
        elif player_choice == "2":
            AI_player = 2
        else:
            print("player_choice should be on these strings: \"1\" | \"2\" | \"both\". Exiting")
            exit()
        
        # --- Main test loop --- #
        while True:
            
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                print("Zero valid actions!")
                exit()

            player = game.current_player
            if player == 1:
                net_to_use = p1_nn
            else:
                net_to_use = p2_nn
            
            # Use neural network or random choice according to the player #
            if (AI_player == 0) or (player == AI_player):
                state = game.generate_state_image()
                action_probs, value_pred = net_to_use.inference(state, False, recurrent_iterations)
                action_probs = softmax(action_probs)
                probs = action_probs.flatten()

                raw_action = np.argmax(probs)
                if not valid_actions_mask[raw_action]:

                    if self.slow:
                        print("AI chose an invalid move. Doing workaround.")
                    
                    probs = probs * valid_actions_mask
                    total = np.sum(probs)

                    if (total != 0): # happens if the network gave 0 probablity to all valid actions and high probability to invalid actions
                        probs /= total

                        max_action = np.argmax(probs)
                        chance_action = np.random.choice(game.num_actions, p=probs)
                        action_i = max_action

                    else:
                        # Problem during learning... using random action instead
                        probs = probs + valid_actions_mask
                        probs /= n_valids
                        action_i = np.random.choice(game.num_actions, p=probs)
                
                else:
                    action_i = raw_action
            
            else:
                # The other player chooses randomly
                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.num_actions, p=probs)

                
            action_coords = np.unravel_index(action_i, game.action_space_shape)
            
            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)
            
            if keep_state_history:
                state = game.generate_state_image()
                game.store_state(state)

            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))

            if (done):
                winner = game.get_winner()
                break
            
        return winner, {}

    def random_vs_random(self, game, keep_state_history=False):
        
        # --- Printing and rendering preparations --- #
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote()
            time.sleep(5)

        # --- Main test loop --- #
        while True:
            
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)
            if (n_valids == 0):
                print("Zero valid actions!")
                exit()
                

            probs = valid_actions_mask/n_valids
            action_i = np.random.choice(game.num_actions, p=probs)

            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)

            if keep_state_history:
                state = game.generate_state_image()
                game.store_state(state)

            action_coords = np.unravel_index(action_i, game.action_space_shape)
            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))

            if (done):
                winner = game.get_winner()
                break

        return winner, {}

    def ttt_vs_AI_with_policy(self, AI_player, nn, recurrent_iterations=1):

        game = tic_tac_toe()

        print("\n")
        while True:

            player = game.current_player
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                    print("Zero valid actions!")
            
            if player == AI_player:

                state = game.generate_state_image()
                action_probs, value_pred = nn.inference(state, False, recurrent_iterations)
                action_probs = softmax(action_probs)
                probs = action_probs.flatten()                    

                raw_action = np.argmax(probs)
                if not valid_actions_mask[raw_action]:
                    
                    if self.debug:
                        print("\ninvalid")
                    
                    
                    probs = probs * valid_actions_mask
                    total = np.sum(probs)

                    if (total != 0):

                        probs /= total

                        probs = np.asarray(probs, dtype=np.float64).astype('float64')
                        probs /= np.sum(probs) # re-normalize to improve precison

                        max_action = np.argmax(probs)
                        chance_action = np.random.choice(game.num_actions, p=probs)
                        action_i = max_action

                    else:
                        # Problem during learning... using random action instead
                        probs = probs + valid_actions_mask
                        probs /= n_valids

                        action_i = np.random.choice(game.num_actions, p=probs)
                
                else:
                    action_i = raw_action


                action_coords = np.unravel_index(action_i, game.action_space_shape)
            else:

                x = input("choose coordenates: ")
                coords = eval(x)
                action_coords = (0, coords[0], coords[1])

            print(game.string_representation())
            done = game.step_function(action_coords)
            

            if (done):
                winner = game.get_winner()
                print(game.string_representation())
                break

            
            return winner
    
    def test_game(self, game_class, game_args): #TODO: Very incomplete
        
        # Plays games at random and displays stats about what terminal states it found, who won, etc...

        game = game_class(*game_args)
        

        for g in range(self.num_games):

            terminal_states_count = 0
            game = game_class(*game_args)

            print("Starting board position:\n")
            print(game.string_representation())

            while not game.is_terminal():

                valid_actions_mask = game.possible_actions().flatten()
                n_valids = np.sum(valid_actions_mask)

                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.num_actions, p=probs)

                action_coords = np.unravel_index(action_i, game.action_space_shape)
    
                done = game.step_function(action_coords)

                print(game.string_representation())

                print("Found " + str(terminal_states_count) + " terminal states.")

        
    
        print("\nFunction incomplete!\n")
        return
    

