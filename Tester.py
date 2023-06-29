import sys
import os, psutil
import gc
import math
import time
import ray

from progress.bar import ChargingBar
from progress.spinner import PieSpinner

import numpy as np

from scipy.special import softmax

import torch
from torch import nn

from Node import Node

from SCS.SCS_Game import SCS_Game
from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Neural_Networks.Torch_NN import Torch_NN


class Tester():

    def __init__(self, recurrent_iters=2, mcts_simulations=800, pb_c_base=19652, pb_c_init=1.25, use_terminal=False, slow=False, print=False, render=False):
        
        self.recurrent_iters = recurrent_iters

        self.mcts_simulations = mcts_simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.use_terminal = use_terminal

        self.slow = slow
        self.print = print

        self.render = render
        if render == True:
            self.slow=True
            # Render is only supported for SCS games
            from SCS.SCS_Renderer import SCS_Renderer
            from RemoteStorage import RemoteStorage

            self.remote_storage = RemoteStorage.remote()
            self.renderer = SCS_Renderer.remote(self.remote_storage)

        self.slow_duration = 2
            

    def set_mcts_parameters(self, mcts_simulations, pb_c_base, pb_c_init, use_terminal):
        self.mcts_simulations = mcts_simulations
        self.pb_c_base = pb_c_base
        self.pb_c_init = pb_c_init
        self.use_terminal = use_terminal

    def set_slow_duration(self, seconds):
        self.slow_duration = seconds
    
# ------------------------------------------------ #
# --------------------- MCTS --------------------- #
# ------------------------------------------------ #

    def run_test_mcts(self, game, network):

        search_start = Node(0)

        num_searches = self.mcts_simulations
        
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
                value = self.evaluate(node, scratch_game, network)
            
            self.backpropagate(search_path, value)

        return self.select_max_action(scratch_game, search_start)
        
    def select_max_action(self, game, node):
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]
        return self.max_action(visit_counts)

    def max_action(self, visit_counts):
        max_pair = max(visit_counts, key=lambda visit_action_pair: visit_action_pair[0])
        return max_pair[1]

    def select_child(self, node):
        _, action, child = max((self.ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child

    def ucb_score(self, parent, child):
        pb_c_base = self.pb_c_base
        pb_c_init = self.pb_c_init

        bias = math.log((parent.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
        pb_c = (math.sqrt(parent.visit_count) / (child.visit_count + 1)) * bias

        prior_score = pb_c * child.prior


        value_score = child.value()
        if parent.to_play == 2:
            value_score = -value_score
        # for player 2 negative values are good

        return prior_score + value_score

    def evaluate(self, node, game, network):
        
        node.to_play = game.get_current_player()
        
        if self.use_terminal and game.is_terminal():
            node.terminal_value = game.get_terminal_value()
            value = node.terminal_value
            return value
        
        state = game.generate_state_image()
        action_probs, predicted_value = network.inference(state, False, self.recurrent_iters)

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
    
    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value	


# ------------------------------------------------ #
# ----------------- TEST METHODS ----------------- #
# ------------------------------------------------ #

    def Test_AI_with_mcts(self, AI_player, game, nn):
        
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote()
            time.sleep(5)

        while True:
            
            player = game.current_player
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                print("Zero valid actions!")
                exit()
            
            if player == AI_player:
                action_i = self.run_test_mcts(game, nn)
            
            
            else:
                # The other player chooses randomly
                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.num_actions, p=probs)


            action_coords = np.unravel_index(action_i, game.action_space_shape)

            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)

            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))

            if (done):
                if self.print:
                    print(game.string_representation())
                winner = game.check_winner()
                break
                

        return winner

    def Test_AI_with_policy(self, AI_player, game, nn):
        
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote()
            time.sleep(5)

        while True:
            
            player = game.current_player
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                print("Zero valid actions!")
                exit()
            
            if player == AI_player:

                state = game.generate_state_image()
                action_probs, value_pred = nn.inference(state, False, self.recurrent_iters)
                probs = action_probs.cpu()[0].numpy().flatten()
            

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

            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))

            if (done):
                winner = game.check_winner()
                break
            
        return winner

    def Test_AI_vs_AI_with_policy(self, game, nn_1, nn_2):
            
            if self.print:
                print("\n")

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))
                self.renderer.render.remote()
                time.sleep(5)
            
            while True:
                                
                player = game.current_player
                valid_actions_mask = game.possible_actions()
                valid_actions_mask = valid_actions_mask.flatten()
                n_valids = np.sum(valid_actions_mask)

                state = game.generate_state_image()
                game.store_state(state)

                if (n_valids == 0):
                        print("Zero valid actions!")
                
                if player == 1:

                    action_probs, value_pred = nn_1.inference(state, False, self.recurrent_iters)
                    probs = action_probs.cpu()[0].numpy().flatten()

                    raw_action = np.argmax(probs)
                    if not valid_actions_mask[raw_action]:

                        probs = probs * valid_actions_mask
                        total = np.sum(probs)
                        if (total != 0):
                            
                            probs /= total

                            probs = np.asarray(probs, dtype=np.float64).astype('float64')
                            probs /= np.sum(probs) # re-normalize to improve precison

                            max_action = np.argmax(probs)
                            chance_action = np.random.choice(game.num_actions, p=probs)
                            action_i = chance_action

                        else:
                            # Problem during learning... using random action instead
                            probs = probs + valid_actions_mask
                            probs /= n_valids

                            action_i = np.random.choice(game.num_actions, p=probs)
                    
                    else:
                        action_i = raw_action
                
                elif player == 2:

                    action_probs, value_pred = nn_2.inference(state, False, self.recurrent_iters)
                    probs = action_probs.cpu()[0].numpy().flatten()

                    raw_action = np.argmax(probs)
                    if not valid_actions_mask[raw_action]:

                        probs = probs * valid_actions_mask
                        total = np.sum(probs)
                        if (total != 0):
                            
                            probs /= total

                            probs = np.asarray(probs, dtype=np.float64).astype('float64')
                            probs /= np.sum(probs) # re-normalize to improve precison

                            max_action = np.argmax(probs)
                            chance_action = np.random.choice(game.num_actions, p=probs)
                            action_i = chance_action

                        else:
                            # Problem during learning... using random action instead
                            probs = probs + valid_actions_mask
                            probs /= n_valids

                            action_i = np.random.choice(game.num_actions, p=probs)
                    
                    else:
                        action_i = raw_action
            

                action_coords = np.unravel_index(action_i, game.action_space_shape)

                if self.print:
                    print(game.string_representation())
                
                if self.slow:
                    time.sleep(self.slow_duration)

                done = game.step_function(action_coords)

                if self.render:
                    ray.get(self.remote_storage.set_item.remote(game))

            
                if (done):
                    winner = game.check_winner()
                    break
                
            return winner

    def ttt_vs_AI_with_policy(self, AI_player, nn):

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
                action_probs, value_pred = nn.inference(state, False, self.recurrent_iters)
                probs = action_probs[0].cpu().numpy()
                probs = probs.flatten()                    

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
                winner = game.check_winner()
                print(game.string_representation())
                break

            
            return winner
    
    def SCS_vs_AI_with_policy(self, AI_player, game_args, nn):              

        game = SCS_Game(*game_args)

        print("\n")
        while True:
            print("\nTurn: " + str(game.current_turn) + "\n")

            player = game.current_player
            valid_actions_mask = game.possible_actions().flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                    print("Zero valid actions!")
            
            if player == AI_player:
                
                state = game.generate_state_image()
                action_probs, value_pred = nn.inference(state, False, self.recurrent_iters)
                probs = action_probs[0].cpu().numpy()
                probs = probs.flatten()

                total = np.sum(probs)

                print("\nValue: ")
                print(value_pred.cpu())

                raw_action = np.argmax(probs)
                if not valid_actions_mask[raw_action]:
                    
                    if self.debug:
                        print("\ninvalid")
                    
                    if (total != 0):
                        probs = probs * valid_actions_mask
                        probs /= total

                        probs = np.asarray(probs, dtype=np.float64).astype('float64')
                        probs /= np.sum(probs) # re-normalize to improve precison

                        max_action = np.argmax(probs)
                        chance_action = np.random.choice(game.num_actions, p=probs)
                        action_i = max_action

                    else:
                        # Problem during learning... using random action instead
                        print("Problem during learning... using random action instead...")
                        probs = probs + valid_actions_mask
                        probs /= n_valids

                        action_i = np.random.choice(game.num_actions, p=probs)
                
                else:
                    action_i = raw_action

                
            else:
                if player == 1:
                    stages = [0,2,3]
                else:
                    stages = [1,4,5]


                invalid = True
                while invalid:
                    print()
                    if game.current_stage == stages[0]:
                        unit_type = input("Type 1 to place a soldier and 2 to place a tank: ")
                        input_coords = input("\nType the coordenates, where to place your unit: ")
                        unit_coords = eval(input_coords)
                        action_i = np.ravel_multi_index([int(unit_type)-1, unit_coords[0]-1, unit_coords[1]-1], game.action_space_shape)
                        if valid_actions_mask[action_i]:
                            invalid = False

                    elif game.current_stage == stages[1]:
                        input_coords = input("Choose the coordenates of the unit, you would like to pick up: ")
                        unit_coords = eval(input_coords)
                        input_dest = input("\nType the coordenates, where to move the unit: ")
                        dest_coords = eval(input_dest)
                        # unraveling the index of the plane 
                        dest_plane = np.ravel_multi_index([(dest_coords[0]-1), (dest_coords[1]-1)], (game.HEIGHT, game.WIDTH)) + 2
                        action_i = np.ravel_multi_index([dest_plane, unit_coords[0]-1, unit_coords[1]-1], game.action_space_shape)
                        if valid_actions_mask[action_i]:
                            invalid = False

                    elif game.current_stage == stages[2]:
                        input_coords= input("Type the coordenates of one of your units: ")
                        unit_coords = eval(input_coords)
                        index_coords = [unit_coords[0]-1, unit_coords[1]-1]
                        fight_answer = input("\nType 1 to fight with this unit and 2 not to fight with it. ")
                        no_fight = int(fight_answer)-1
                        if no_fight:    
                            action_i = np.ravel_multi_index([6 + game.HEIGHT*game.WIDTH, index_coords[0], index_coords[1]], game.action_space_shape)
                        else:
                            direction = input("\nChoose a direction to fight towards (N,S,E,W): ")
                            if direction == "N" or direction == "n":
                                plane_offset = 0
                                dest_coords= ((index_coords[0])-1, (index_coords[1]))
                            elif direction == "S" or direction == "s":
                                dest_coords= ((index_coords[0])+1, (index_coords[1]))
                                plane_offset = 1
                            elif direction == "E" or direction == "e":
                                dest_coords= ((index_coords[0]), (index_coords[1])+1)
                                plane_offset = 2
                            elif direction == "W" or direction == "w":
                                dest_coords = ((index_coords[0]), (index_coords[1])-1)
                                plane_offset = 3

                            dest_plane = 2 + game.HEIGHT*game.WIDTH + plane_offset
                            action_i = np.ravel_multi_index([dest_plane , index_coords[0], index_coords[1]], game.action_space_shape)
                        if valid_actions_mask[action_i]:
                            invalid = False


            action_coords = np.unravel_index(action_i, game.action_space_shape)

            print(game.string_representation())
            time.sleep(self.slow_duration)

            done = game.step_function(action_coords)


            if (done):
                print(game.string_representation())
                winner = game.check_winner()
                break

        return winner
          
    def SCS_vs_AI_with_mcts(self, AI_player, game_args, nn):

        game = SCS_Game(*game_args)

        print("\n")
        while True:
            print("\nTurn: " + str(game.current_turn) + "\n")

            player = game.current_player
            valid_actions_mask = game.possible_actions().flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                    print("Zero valid actions!")
            
            if player == AI_player:
                action_i = self.run_test_mcts(game, nn)
                
            else:
                if player == 1:
                    stages = [0,2,3]
                else:
                    stages = [1,4,5]


                invalid = True
                while invalid:
                    print()
                    if game.current_stage == stages[0]:
                        unit_type = input("Type 1 to place a soldier and 2 to place a tank: ")
                        input_coords = input("\nType the coordenates, where to place your unit: ")
                        unit_coords = eval(input_coords)
                        action_i = np.ravel_multi_index([int(unit_type)-1, unit_coords[0]-1, unit_coords[1]-1], game.action_space_shape)
                        if valid_actions_mask[action_i]:
                            invalid = False

                    elif game.current_stage == stages[1]:
                        input_coords = input("Choose the coordenates of the unit, you would like to pick up: ")
                        unit_coords = eval(input_coords)
                        input_dest = input("\nType the coordenates, where to move the unit: ")
                        dest_coords = eval(input_dest)
                        # unraveling the index of the plane 
                        dest_plane = np.ravel_multi_index([(dest_coords[0]-1), (dest_coords[1]-1)], (game.HEIGHT, game.WIDTH)) + 2
                        action_i = np.ravel_multi_index([dest_plane, unit_coords[0]-1, unit_coords[1]-1], game.action_space_shape)
                        if valid_actions_mask[action_i]:
                            invalid = False

                    elif game.current_stage == stages[2]:
                        input_coords= input("Type the coordenates of one of your units: ")
                        unit_coords = eval(input_coords)
                        index_coords = [unit_coords[0]-1, unit_coords[1]-1]
                        fight_answer = input("\nType 1 to fight with this unit and 2 not to fight with it. ")
                        no_fight = int(fight_answer)-1
                        if no_fight:    
                            action_i = np.ravel_multi_index([6 + game.HEIGHT*game.WIDTH, index_coords[0], index_coords[1]], game.action_space_shape)
                        else:
                            direction = input("\nChoose a direction to fight towards (N,S,E,W): ")
                            if direction == "N" or direction == "n":
                                plane_offset = 0
                                dest_coords= ((index_coords[0])-1, (index_coords[1]))
                            elif direction == "S" or direction == "s":
                                dest_coords= ((index_coords[0])+1, (index_coords[1]))
                                plane_offset = 1
                            elif direction == "E" or direction == "e":
                                dest_coords= ((index_coords[0]), (index_coords[1])+1)
                                plane_offset = 2
                            elif direction == "W" or direction == "w":
                                dest_coords = ((index_coords[0]), (index_coords[1])-1)
                                plane_offset = 3

                            dest_plane = 2 + game.HEIGHT*game.WIDTH + plane_offset
                            action_i = np.ravel_multi_index([dest_plane , index_coords[0], index_coords[1]], game.action_space_shape)

                        if valid_actions_mask[action_i]:
                            invalid = False


            action_coords = np.unravel_index(action_i, game.action_space_shape)

            print(game.string_representation())

            done = game.step_function(action_coords)

            if (done):
                print(game.string_representation())
                winner = game.check_winner()
                break

            
        return winner

    def random_vs_random(self, game):
        
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.set_item.remote(game))
            self.renderer.render.remote()
            time.sleep(5)

        while True:
            
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)
            if (n_valids == 0):
                print("Zero valid actions!")
                exit()
                
            '''
            print("\n||||||||||||||||||||||||||||||||||||||||")
            for i in range(len(valid_actions_mask)):
                if valid_actions_mask[i] == 1:
                    coords = np.unravel_index(i, game.action_space_shape)
                    print()
                    print(game.string_action(coords))

            print("\n||||||||||||||||||||||||||||||||||||||||\n\n")     
            '''

            probs = valid_actions_mask/n_valids
            action_i = np.random.choice(game.num_actions, p=probs)

            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)

            action_coords = np.unravel_index(action_i, game.action_space_shape)
            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.set_item.remote(game))

            if (done):
                winner = game.check_winner()
                break

        return winner

    def test_game(self, game_class, game_args): #TODO: Incomplete
        
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
