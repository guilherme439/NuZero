import sys
import os
import psutil
import time
import random
import math
import pickle
import ray
import copy
import argparse
import glob
import re
import torch

import numpy as np
import matplotlib.pyplot as plt

from ray.runtime_env import RuntimeEnv
from scipy.special import softmax

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network as MLP_Net

from Neural_Networks.ConvNet import ConvNet
from Neural_Networks.ResNet import ResNet
from Neural_Networks.RecurrentNet import RecurrentNet

from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from AlphaZero import AlphaZero
from Tester import Tester
from RemoteTester import RemoteTester

from Gamer import Gamer
from ReplayBuffer import ReplayBuffer
from RemoteStorage import RemoteStorage

from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.RandomAgent import RandomAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent

from TestManager import TestManager

from progress.bar import ChargingBar
from Utils.Progress_Bars.PrintBar import PrintBar

from Utils.general_utils import *

from Utils.Caches.KeylessCache import KeylessCache
from Utils.Caches.DictCache import DictCache

import Interactive

def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--interactive", action='store_true',
        help="Create a simple training setup interactivly"
    )
    exclusive_group.add_argument(
        "--training-preset", type=int,
        help="Choose one of the preset training setups"
    )
    exclusive_group.add_argument(
        "--testing-preset", type=int,
        help="Choose one of the preset testing setups"
    )
    exclusive_group.add_argument(
        "--debug", type=int,
        help="Choose one of the debug modes"
    )
    parser.add_argument(
        "--log-driver", action='store_true',
        help="log_to_driver=True"
    )
    

    args = parser.parse_args()

    print("\n\nCUDA Available: " + str(torch.cuda.is_available()))

    log_to_driver = False
    if args.log_driver:
        log_to_driver = True

    if args.training_preset is not None:
        
        ##############################################################################################################
        # ---------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------   TRAINING-PRESETS   ------------------------------------------ #
        # ---------------------------------------------------------------------------------------------------------- #
        ##############################################################################################################

        match args.training_preset:

            case 0: # Tic_tac_toe example
                game_class = tic_tac_toe
                game_args_list = [[]]
                game = game_class()

                alpha_config_path="Configs/Config_Files/Training/small_test_training_config.yaml"
                search_config_path="Configs/Config_Files/Search/test_search_config.yaml"

                state_set = create_r_unbalanced_state_set(game)

                model = MLP_Net(out_features=9)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()
                
            case 1: # SCS_Example
                
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/r_unbalanced_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Config_Files/Training/small_test_training_config.yaml"
                search_config_path="Configs/Config_Files/Search/test_search_config.yaml"

                state_set = create_r_unbalanced_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()

                
            ###############  CUSTOM PRESETS  ###################
                
            case 2:
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/r_unbalanced_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Training/small_dummy_training_config.yaml"
                search_config_path="Configs/Search/dummy_search_config.yaml"

                state_set = create_r_unbalanced_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=12, policy_head="conv", value_head="reduce", hex=True)

                initialize_parameters(model)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()


            case 3:
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/r_unbalanced_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Training/small_test_training_config.yaml"
                search_config_path="Configs/Search/test_search_config.yaml"


                state_set = create_r_unbalanced_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=12, policy_head="conv", value_head="reduce", hex=True)

                initialize_parameters(model)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()



            case 4:
                # Define your setup here
                exit()

            case _:
                print("\n\nUnknown training preset")
                exit()

    
    elif args.testing_preset is not None:

        ##############################################################################################################
        # ---------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------   TESTING-PRESETS   ------------------------------------------- #
        # ---------------------------------------------------------------------------------------------------------- #
        ##############################################################################################################

        match args.testing_preset:
            case 0: # Render Example
                game_class = SCS_Game
                game_args = ["Games/SCS/Game_configs/randomized_config_5.yml"]

                test_config = "Configs/Testing/Examples/render_config.yaml"
                test_manager = TestManager(game_class, game_args)
                results = test_manager.test_from_config(test_config_path=test_config)



            case 1: # Extrapolation Test Example
                game_class = SCS_Game
                game_args = ["Games/SCS/Game_configs/randomized_config_5.yml"]

                test_config = "Configs/Testing/Examples/extrapolation_config.yaml"
                test_manager = TestManager(game_class, game_args)
                results = test_manager.test_from_config(test_config_path=test_config)
                
                #'''
                x, y = zip(*results)
                plt.plot(x, y)
                plt.show()
                #'''


            ###############  CUSTOM PRESETS  ###################

            case 2:
                pass

            case _:
                raise Exception("Unknown testing preset.")

        
        
            
    elif args.debug is not None:
        match args.debug:
            
            case 0: # Test initialization
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config_5.yml"]
                game = game_class(*game_args)


                #nn, search_config = load_trained_network(game, "adam_se_mse_mirror", 130)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="dense")

                print(model)
                #'''
                for name, param in model.named_parameters():
                    #print(name)
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.75)
                    
                #'''
                nn = Torch_NN(model)

                
                play_actions = 9
                for _ in range(play_actions):
                    valid_actions_mask = game.possible_actions()
                    valid_actions_mask = valid_actions_mask.flatten()
                    n_valids = np.sum(valid_actions_mask)
                    probs = valid_actions_mask/n_valids
                    action_i = np.random.choice(game.num_actions, p=probs)
                    action_coords = np.unravel_index(action_i, game.action_space_shape)
                    game.step_function(action_coords)

                
                state = game.generate_state_image()

                policy, value = nn.inference(state, False, 30)
                

                print("\n\n")
                #print(policy)
                print("\n\n")
                print(torch.sum(policy))
                print(value)
                print("\n\n----------------\n\n")
                
                all_weights = torch.Tensor().cpu()
                for param in nn.get_model().parameters():
                    #print(param)
                    all_weights = torch.cat((all_weights, param.clone().detach().flatten().cpu()), 0) 

                print(all_weights)

                
                

            case 1:
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config_5.yml"]
                game = game_class(*game_args)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")

                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,20,100], gamma=0.1)

                file_name = "optimizer"
                torch.save(optimizer.state_dict(), file_name)
                optimizer_sd = torch.load(file_name)
                

                print("\n\n\n\n\n")

                file_name = "scheduler"
                torch.save(scheduler.state_dict(), file_name)
                loaded_data = torch.load(file_name)
                print(loaded_data)
                os.remove(file_name)



    elif args.interactive:
        Interactive.start_interative()


    
    return

def initialize_parameters(model):
    for name, param in model.named_parameters():
        if ".weight" not in name:
            #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
            torch.nn.init.xavier_uniform_(param)

##########################################################################
# ----------------------------               --------------------------- #
# -----------------------       STATE SETS       ----------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def create_mirrored_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
    game.reset_env()
    game.set_simple_game_state(9, [1], [(0,1)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1,1,1], [(0,1),(1,1),(0,0)], [2,2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1], [(4,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(9, [1,1,1,1], [(0,1),(0,1),(0,0),(0,0)], [2,2,1,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1,1,1], [(4,3),(3,3),(4,4)], [1,1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(9, [1], [(4,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_unbalanced_state_set(game):
    renderer = SCS_Renderer()

    state_set = []

    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,1)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1,1], [(0,1),(1,1),(0,0)], [2,2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(2,2),(2,1)], [2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(3,0)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_r_unbalanced_state_set(game):
    renderer = SCS_Renderer()

    state_set = []

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,2)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(0,1),(4,3)], [2,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    
    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(2,3),(3,3)], [1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1,1], [(1,4),(2,2),(2,3)], [1,1,2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,4)], [1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1,1], [(4,3),(4,3)], [1,1])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

def create_solo_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,0)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,3)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,2)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(2,3)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(2,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------       RAY       -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def start_ray_local(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    context = ray.init(log_to_driver=log_to_driver)
    return context

def start_ray_local_cluster(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					pip="./requirements.txt"
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context

def start_ray_rnl(log_to_driver=False):
    print("\n\n--------------------------------\n\n")

    '''
    env_vars={"CUDA_VISIBLE_DEVICES": "-1",
            "LD_LIBRARY_PATH": "$NIX_LD_LIBRARY_PATH"
            }
    '''

    runtime_env=RuntimeEnv \
					(
					working_dir="/mnt/cirrus/users/5/2/ist189452/TESE/NuZero",
					pip="./requirements.txt",
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=log_to_driver)
    return context


# ------------------------

if __name__ == "__main__":
    main()