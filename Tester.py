import math
import time

import ray
import torch
import numpy as np


from progress.bar import ChargingBar
from progress.spinner import PieSpinner

from Utils.PrintBar import PrintBar


from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Agents.Generic.MctsAgent import MctsAgent




class Tester():

    def __init__(self, slow=False, print=False, render=False):
        #torch.multiprocessing.set_sharing_strategy('file_system')

        self.slow = slow
        self.print = print

        self.render = render
        if render == True:
            self.slow=True
            # Render is only supported for SCS games
            from SCS.SCS_RemoteRenderer import SCS_RemoteRenderer
            from RemoteStorage import RemoteStorage

            self.remote_storage = RemoteStorage.remote(window_size=1)
            self.renderer = SCS_RemoteRenderer.remote(self.remote_storage)

        self.slow_duration = 2

    def set_slow_duration(self, seconds):
        self.slow_duration = seconds
    
# ------------------------------------------------ #
# ----------------- TEST METHODS ----------------- #
# ------------------------------------------------ #

    def Test_using_agents(self, game, p1_agent, p2_agent, keep_state_history=False):

        
        # --- Printing and rendering preparations --- #
        if self.print:
            print("\n")

        if self.render:
            ray.get(self.remote_storage.store.remote(game))
            self.renderer.render.remote(player_unit_images=True)
            time.sleep(3)

        
        # --- Main test loop --- #
        while True:
            
            valid_actions_mask = game.possible_actions().flatten()

            player = game.current_player
            
            if (player == 1):
                current_agent = p1_agent
                opponent_agent = p2_agent
            else:
                current_agent = p2_agent
                opponent_agent = p1_agent

            
            action_coords = current_agent.choose_action(game)
            
            action_i = game.get_action_index(action_coords)
            if not valid_actions_mask[action_i]:
                print("invalid agent action\n")
                exit()

            # When using MctsAgents, if we want to keep the subtree,
            # we need to run the agent in the opponent's turn,
            # in order to mantain the search tree updated
            if isinstance(opponent_agent, MctsAgent):
                if opponent_agent.keep_subtree:
                    _ = opponent_agent.update_subtree(game, action_i)
                
            if self.print:
                print(game.string_representation())

            if self.slow:
                time.sleep(self.slow_duration)
            
            if keep_state_history:
                state = game.generate_state_image()
                game.store_state(state)

            done = game.step_function(action_coords)

            if self.render:
                ray.get(self.remote_storage.store.remote(game))

            if (done):
                if self.print:
                    print(game.string_representation())
                winner = game.get_winner()
                break
            
            
        return winner, {}

    def ttt_vs_agent(self, user_player, agent):

        game = tic_tac_toe()

        print("\n")
        while True:

            player = game.current_player
            valid_actions_mask = game.possible_actions()
            valid_actions_mask = valid_actions_mask.flatten()
            n_valids = np.sum(valid_actions_mask)

            if (n_valids == 0):
                    print("Zero valid actions!")
            
            if player != user_player:
                action_coords = agent.choose_action(game)
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
    

