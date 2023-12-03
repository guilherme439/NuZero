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

from progress.bar import ChargingBar
from Utils.PrintBar import PrintBar

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network as MLP_Net

from Neural_Networks.Orthogonal.Simple_Conv_Network import Simple_Conv_Network as Ort_ConvNet
from Neural_Networks.Orthogonal.ResNet import ResNet as Ort_ResNet
from Neural_Networks.Orthogonal.dt_neural_network import dt_net_2d as Ort_DTNet
from Neural_Networks.Orthogonal.dt_neural_network import dt_net_recall_2d as Ort_DTNet_recall

from Neural_Networks.Hexagonal.ConvNet import ConvNet as Hex_ConvNet
from Neural_Networks.Hexagonal.ResNet import ResNet as Hex_ResNet
from Neural_Networks.Hexagonal.RecurrentNet import RecurrentNet as Hex_RecurrentNet

from SCS.SCS_Game import SCS_Game
from SCS.SCS_Renderer import SCS_Renderer

from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Configs.Training_Config import Training_Config
from Configs.Search_Config import Search_Config

from AlphaZero import AlphaZero
from Tester import Tester
from RemoteTester import RemoteTester

from Utils.stats_utilities import *

from Gamer import Gamer
from ReplayBuffer import ReplayBuffer
from RemoteStorage import RemoteStorage

from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.RandomAgent import RandomAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent

from TestManager import TestManager

from Utils.Caches.KeylessCache import KeylessCache


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
    parser.add_argument(
        "--name", type=str, default="",
        help="Change the name of network trained with the preset"
    )
    parser.add_argument(
        "--log-driver", action='store_true',
        help="log_to_driver=True"
    )
    exclusive_group.add_argument(
        "--debug", type=int,
        help="Choose one of the debug modes"
    )
    exclusive_group.add_argument(
        "--testing-preset", type=int,
        help="Choose one of the preset testing setups"
    )

    args = parser.parse_args()

    print("CUDA: " + str(torch.cuda.is_available()))

               
    if args.training_preset is not None:
        log_to_driver = False
        if args.log_driver:
            log_to_driver = True

        ##############################################################################################################
        # ---------------------------------------------------------------------------------------------------------- #
        # ------------------------------------------   TRAINING-PRESETS   ------------------------------------------ #
        # ---------------------------------------------------------------------------------------------------------- #
        ##############################################################################################################

        match args.training_preset:
            case 0: # Tic_tac_toe example
                game_class = tic_tac_toe
                game_args = []
                game = game_class(*game_args)

                alpha_config_path="Configs/Config_Files/Training/test_training_config.ini"
                search_config_path="Configs/Config_Files/Search/ttt_search_config.ini"

                network_name = "ttt_example_net"

                ################################################

                if args.name is not None and args.name != "":
                    network_name = args.name

                #num_actions = game.get_num_actions()
                #model = MLP_Net(num_actions)
                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Ort_DTNet(in_channels, policy_channels, 64, 1)

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
                alpha_zero.run()

            case 1: # Continue training
                
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/unbalanced_config.yml"]
                game = game_class(*game_args)

                trained_network_name = "unbalanced_new_config_continue"
                continue_network_name = "unbalanced_new_config_continue" # new network can have the same name as the previous
                use_same_configs = True

                # In case of not using the same configs define the new configs to use like this
                new_train_config_path="Configs/Config_Files/Training/local_training_config.ini"
                new_search_config_path="Configs/Config_Files/Search/local_search_config.ini"

                ################################################

                state_set = None
                state_set = create_unbalanced_state_set(game)


                print("\n")
                context = start_ray_local(log_to_driver)
                continue_training(game_class, game_args, trained_network_name, continue_network_name, \
                                  use_same_configs, new_train_config_path, new_search_config_path, state_set)

            case 2:
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config.yml"]
                game = game_class(*game_args)

                alpha_config_path="Configs/Config_Files/Training/a1_training_config.ini"
                search_config_path="Configs/Config_Files/Search/local_search_config.ini"

                network_name = "local_net_test"

                ################################################

                state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = Hex_ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="dense")

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.75)
                    
                #'''

                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path, state_set=state_set)
                alpha_zero.run()

            case 3:
                game_class = SCS_Game
                game_args_list = [ ["SCS/Game_configs/solo_soldier_config_4.yml"],
                                   ["SCS/Game_configs/solo_soldier_config_5.yml"], 
                                   ["SCS/Game_configs/solo_soldier_config_9.yml"] ]
                
                game = game_class(*game_args_list[1])

                alpha_config_path="Configs/Config_Files/Training/a1_training_config.ini"
                search_config_path="Configs/Config_Files/Search/local_search_config.ini"

                network_name = "local_net_test"

                ################################################

                state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_RecurrentNet(in_channels, policy_channels, 300, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = Hex_ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="reduce")

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.8)
                    
                #'''

                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, model, network_name, alpha_config_path, search_config_path, state_set=state_set)
                alpha_zero.run()


            case 4:
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config.yml"]
                game = game_class(*game_args)

                alpha_config_path="Configs/Config_Files/Training/a2_training_config.ini"
                search_config_path="Configs/Config_Files/Search/local_search_config.ini"

                network_name = "local_net_test"

                ################################################

                state_set = create_mirrored_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = Hex_ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="dense")

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.75)
                    
                #'''

                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path, state_set=state_set)
                alpha_zero.run()
                

            case 5: 
                
                game_class = SCS_Game
                game_args_list = [ ["SCS/Game_configs/solo_soldier_config_4.yml"],
                                   ["SCS/Game_configs/solo_soldier_config_5.yml"], 
                                   ["SCS/Game_configs/solo_soldier_config_9.yml"] ]
                
                game = game_class(*game_args_list[1])

                alpha_config_path="Configs/Config_Files/Training/test_training_config.ini"
                search_config_path="Configs/Config_Files/Search/test_search_config.ini"

                network_name = "local_net_test"

                ################################################

                state_set = create_solo_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_RecurrentNet(in_channels, policy_channels, 64, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = Hex_ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="reduce")

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.85)
                    
                #'''

                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, model, network_name, alpha_config_path, search_config_path, state_set=state_set)
                alpha_zero.run()




            case 6:
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
            case 0: # Tic Tac Toe Example
                ray.init()

                number_of_testers = 5

                game_class = tic_tac_toe
                game_args = []
                method = "mcts"

                # testing options
                num_games = 200
                AI_player = "1"
                recurrent_iterations = 2

                # network options
                net_name = "best_ttt_config"
                model_iteration = 600

                ################################################
                
                game = game_class(*game_args)
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                
                test_loop(number_of_testers, method, num_games, game_class, game_args, AI_player, search_config, nn, recurrent_iterations, False)

            case 1: # Render Game
                rendering_mode = "interactive"  # passive | interactive

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config_4.yml"]
                game = game_class(*game_args)

                # network options
                net_name = "new_solo"
                model_iteration = 1350
                recurrent_iterations = 4


                nn, search_config = load_trained_network(game, net_name, model_iteration)
                
                # Agents
                mcts_agent = MctsAgent(search_config, nn, recurrent_iterations, "disabled")
                policy_agent = PolicyAgent(nn, recurrent_iterations)
                random_agent = RandomAgent()
                goal_agent = GoalRushAgent(game)
                p1_agent = random_agent
                p2_agent = policy_agent
                
                ################################################

                
                if rendering_mode == "passive":
                    tester = Tester(render=True)
                elif rendering_mode == "interactive":
                    tester = Tester(print=True)

                
                winner, _ = tester.Test_using_agents(game, p1_agent, p2_agent, keep_state_history=False)
                

                if winner == 0:
                    winner_text = "Draw!"
                else:
                    winner_text = "Player " + str(winner) + " won!"
                
                print("\n\nLength: " + str(game.get_length()) + "\n")
                print(winner_text)
                
                if rendering_mode == "interactive":
                    time.sleep(0.5)

                    renderer = SCS_Renderer()
                    renderer.analyse(game)

            case 2: # Statistics for multiple games
                num_testers = 5
                num_games = 200

                game_class = SCS_Game
                game_config = "SCS/Game_configs/solo_soldier_config_10.yml"
                game_args = [game_config]
                game = game_class(*game_args)

                # network options
                net_name = "new_solo"
                model_iteration = 1350
                recurrent_iterations = 10

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                
                # Agents
                mcts_agent = MctsAgent(search_config, nn, recurrent_iterations, "keyless", 1000)
                policy_agent = PolicyAgent(nn, recurrent_iterations)
                random_agent = RandomAgent()
                goal_agent = GoalRushAgent(game)
                p1_agent = random_agent
                p2_agent = policy_agent

                ################################################
                print("\n")
                print(game_args)
                print("Testing with " + str(recurrent_iterations) + " recurrent iterations.\n")

                test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)

                time.sleep(1)

            case 3: # Graphs for several network checkpoints
                ray.init()

                num_testers = 5
                num_games = 200

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config_10.yml"]
                game = game_class(*game_args)

                recurrent_iterations = 10

                # network options
                net_name = "new_solo"


                # Test Manager configuration
                shared_storage = RemoteStorage.remote(window_size=1)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                

                #---
                min = 650
                max = 1350
                step = 50
                network_iterations_list = range(min,max+1,step)
                
                name = net_name + "_10x10_" + str(min) + "-" + str(max) + "_10_iterarions"
                figpath = "Graphs/networks/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, recurrent_iterations, "keyless", 500)
                #policy_agent = PolicyAgent(nn, recurrent_iterations)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()


                p1_wr_list = []
                p2_wr_list = []
                for net_iter in network_iterations_list:
                    print("\n\n\nTesting network n." + str(net_iter) + "\n")
                    nn, search_config = load_trained_network(game, net_name, net_iter)
                    shared_storage.store.remote(nn)
                    p1_agent = RandomAgent()
                    p2_agent = PolicyAgent(nn, recurrent_iterations)
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(network_iterations_list, p1_wr_list, label = "P1")
                plt.plot(network_iterations_list, p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

            case 4: # Graphs for several recurrent iterations (extrapolation testing)
                ray.init()

                num_testers = 5
                num_games = 200

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config_10.yml"]
                game = game_class(*game_args)


                # network options
                net_name = "new_solo"
                model_iteration = 1350

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                

                #---
                min = 0
                max = 20
                step = 1
                recurrent_iterations_list = range(min,max+1,step)
                
                name = "10x10_" + str(min) + "-" + str(max) + "-iterations_" + net_name + "_" + str(model_iteration)
                figpath = "Graphs/iterations/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "keyless")
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()


                p1_wr_list = []
                p2_wr_list = []
                for rec_iter in recurrent_iterations_list:
                    p1_agent = RandomAgent()
                    p2_agent = PolicyAgent(nn, rec_iter)
                    print("\n\n\nTesting with " + str(rec_iter) + " iterations\n")
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(recurrent_iterations_list, p1_wr_list, label = "P1")
                plt.plot(recurrent_iterations_list, p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

            

            case 5: # Graphs for several games (can be used to compared performance with board size for example)
                ray.init()

                game = SCS_Game("SCS/Game_configs/solo_soldier_config.yml")

                num_testers = 5
                num_games = 200

                # network options
                net_name = "new_solo"
                model_iteration = 1250
                recurrent_iterations = 7

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn) 
                
                # Game settings
                game_class = SCS_Game
                configs_list = ["SCS/Game_configs/solo_soldier_config_5.yml",
                                "SCS/Game_configs/solo_soldier_config_6.yml",
                                "SCS/Game_configs/solo_soldier_config_7.yml",
                                "SCS/Game_configs/solo_soldier_config_8.yml",
                                "SCS/Game_configs/solo_soldier_config_9.yml",
                                "SCS/Game_configs/solo_soldier_config_10.yml",
                                "SCS/Game_configs/solo_soldier_config_11.yml",
                                "SCS/Game_configs/solo_soldier_config_12.yml",
                                "SCS/Game_configs/solo_soldier_config_13.yml",
                                "SCS/Game_configs/solo_soldier_config_14.yml",
                                "SCS/Game_configs/solo_soldier_config_15.yml"]
                            
                
                
                name = "5x5_to_15x15_7-iterations_" + net_name + "_" + str(model_iteration)
                figpath = "Graphs/sizes/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "per_game")
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()

                

                p1_wr_list = []
                p2_wr_list = []
                for config in configs_list:
                    game_args = [config]
                    test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                    p1_agent = RandomAgent()
                    p2_agent = PolicyAgent(nn, recurrent_iterations)
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(range(len(configs_list)), p1_wr_list, label = "P1")
                plt.plot(range(len(configs_list)), p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

            case 6: # Graphs for sequence of games and iterations
                ray.init()

                game = SCS_Game("SCS/Game_configs/solo_soldier_config.yml")

                num_testers = 5
                num_games = 200

                # network options
                net_name = "new_solo"
                model_iteration = 1250

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn) 
                
                # Game settings
                game_class = SCS_Game
                configs_list = ["SCS/Game_configs/solo_soldier_config_4.yml",
                                "SCS/Game_configs/solo_soldier_config_5.yml",
                                "SCS/Game_configs/solo_soldier_config_6.yml",
                                "SCS/Game_configs/solo_soldier_config_7.yml",
                                "SCS/Game_configs/solo_soldier_config_8.yml",
                                "SCS/Game_configs/solo_soldier_config_9.yml",
                                "SCS/Game_configs/solo_soldier_config_10.yml",
                                "SCS/Game_configs/solo_soldier_config_11.yml",
                                "SCS/Game_configs/solo_soldier_config_12.yml",
                                "SCS/Game_configs/solo_soldier_config_13.yml",
                                "SCS/Game_configs/solo_soldier_config_14.yml",
                                "SCS/Game_configs/solo_soldier_config_15.yml"]
                            
                #---
                min = 4
                max = 15
                step = 1
                recurrent_iterations_list = list(range(min,max+1,step))
                
                name = str(min) + "-" + str(max) + "_increasing_size_" + net_name + "_" + str(model_iteration)
                figpath = "Graphs/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "per_game")
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()

                

                p1_wr_list = []
                p2_wr_list = []
                for i in range(len(configs_list)):
                    game_args = [configs_list[i]]
                    rec_iter = recurrent_iterations_list[i]
                    test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                    p1_agent = RandomAgent()
                    p2_agent = PolicyAgent(nn, rec_iter)
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(range(len(configs_list)), p1_wr_list, label = "P1")
                plt.plot(range(len(configs_list)), p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()


            
                

            case _:
                print("Unknown testing preset.")
                return


    elif args.debug is not None:
        match args.debug:
            
            case 0: # Test initialization
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config.yml"]
                game = game_class(*game_args)


                #nn, search_config = load_trained_network(game, "adam_se_mse_mirror", 130)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu")
                #model = Hex_ResNet(in_channels, policy_channels, num_filters=256, num_blocks=20, policy_head="conv", value_head="dense")

                print(model)
                #'''
                for name, param in model.named_parameters():
                    #print(name)
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param, gain=0.70)
                    
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
                import more_itertools

                list_of_tuples = [("lala", (933,1), 1),
                                  ("lawe", (20,0), 0),
                                  ("weala", (13,1), 3),
                                  ("lretla", (2243,2), 0),
                                  ("lewrla", (234352,3), 1),
                                  ("xladsala", (2003,7), 3),
                                  ("lfaflefrewe", (2857233,2), 2)]
                
                s = more_itertools.bucket(list_of_tuples, key=lambda x: x[2]) 
                for key in sorted(s):
                    print(key)
                    print(list(s[key]))
                

                
                



    elif args.interactive:
        print("\nStarted interactive mode!\n")
        
        mode_answer = input("\nWhat do you wish to do?(insert the number)\
                             \n 1 -> Train a network\
                             \n 2 -> Test a trained network\
                             \n 3 -> Image creation (WIP)\
                             \n\nNumber: ")
        
        match int(mode_answer):
            case 1:
                training_mode()
            case 2:
                testing_mode()
            case 3:
                images_mode()

            case _:
                print("Option unavailable")    

    
    return

def create_mirrored_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
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

def create_solo_state_set(game):
    renderer = SCS_Renderer()

    state_set = []
    game.set_simple_game_state(7, [1], [(0,0)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(3,0)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(1,2)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(4,2)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(0,3)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)

    game.reset_env()
    game.set_simple_game_state(7, [1], [(2,4)], [2])
    state_set.append(game.generate_state_image())
    #renderer.display_board(game)
    

    game.reset_env()
    return state_set

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------   INTERACTIVE   -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def images_mode():
    return

def testing_mode():
    game_class, game_args = choose_game()
    game = game_class(*game_args)
    game_name = game.get_name()

    if game_name == "Tic_Tac_Toe":
        test_mode_answer = "2"
    else:
        test_mode_answer = input("\nSelect what kind of testing you wish to do.(1 or 2)\
                                \n1 -> Visualize a game\
                                \n2 -> Take statistics from playing many games\n\n")
        
    if test_mode_answer == "1":
        rendering_mode_answer = input("\nDo you wish to render a game while it is being played or analyse a game after it is played?.(1 or 2)\
                                    \n1 -> Render game\
                                    \n2 -> Analyse game\n\n")
        if rendering_mode_answer == "1":
            rendering_mode = "passive"
        elif rendering_mode_answer == "2":
            rendering_mode = "interative"
        else:
            print("\nBad answer.")
            exit()

        
        method = choose_method()
        net_name, model_iteration, recurrent_iterations = choose_trained_network()

        player_answer = input("\nWhat player will the AI take?(1 or 2)\
                            \n1 -> Player 1\
                            \n2 -> Player 2\n\n")
        
        AI_player = player_answer

        ################################################

        game = game_class(*game_args)
        nn, search_config = load_trained_network(game, net_name, model_iteration)
        
        if rendering_mode == "passive":
            tester = Tester(render=True)
        elif rendering_mode == "interactive":
            tester = Tester(print=True)

        if method == "mcts":
            winner, _ = tester.Test_AI_with_mcts(AI_player, search_config, game, nn, use_state_cache=False, recurrent_iterations=recurrent_iterations)
        elif method == "policy":
            winner, _ = tester.Test_AI_with_policy(AI_player, game, nn, recurrent_iterations=recurrent_iterations)
        elif method == "random":
            winner, _ = tester.random_vs_random(game)

        if winner == 0:
            winner_text = "Draw!"
        else:
            winner_text = "Player " + str(winner) + " won!"
        
        print("\n\nLength: " + str(game.get_length()) + "\n")
        print(winner_text)
        
        if rendering_mode == "interactive":
            time.sleep(0.5)

            renderer = SCS_Renderer()
            renderer.analyse(game)

    elif test_mode_answer == "2":
        
        method = choose_method()
        net_name, model_iteration, recurrent_iterations = choose_trained_network()

        player_answer = input("\nWhat player will the AI take?(1 or 2)\
                            \n1 -> Player 1\
                            \n2 -> Player 2\n\n")
        AI_player = player_answer

        num_games = int(input("\nHow many games you wish to play?"))
        number_of_testers = int(input("\nHow many processes/actors you wish to use?"))

        ################################################
        ray.init()

        game = game_class(*game_args)
        nn, search_config = load_trained_network(game, net_name, model_iteration)
        
        test_loop(number_of_testers, method, num_games, game_class, game_args, AI_player, search_config, nn, recurrent_iterations, False)

def training_mode():
    game_class, game_args = choose_game()
    game = game_class(*game_args)
    game_folder_name = game.get_name()

    continue_answer = input("\nDo you wish to continue training a previous network or train a new one?(1 or 2)\
                                \n1 -> Continue training\
                                \n2 -> New Network\
                                \n\nNumber: ")
    
    if continue_answer == "1":
        continuing = True
    elif continue_answer == "2":
        continuing = False
    else:
        print("Unknown answer.")
        exit()
                
    if continuing:
        trained_network_name = input("\nName of the existing network: ")
        continue_network_name = trained_network_name
        new_name_answer = input("\nDo you wish to continue with the new name?(y/n)")
        if new_name_answer == "y":
            continue_network_name = input("\nNew name: ")

        configs_answer = input("\nDo you wish to use the same previous configs?(y/n)")
        if configs_answer == "y":
            use_same_configs = True
            new_alpha_config_path = ""
            new_search_config_path = ""
        else:
            use_same_configs = False
            print("\nYou will new to provide new configs.")
            new_alpha_config_path = input("\nAlpha config path: ")
            new_search_config_path = input("\nSearch config path: ")

        continue_training(game_class, game_args, trained_network_name, continue_network_name, \
                                use_same_configs, new_alpha_config_path, new_search_config_path)
        
    else:
        invalid = True
        network_name = input("\nName of the new network to train: ")
        while invalid:
            model_folder_path = game_folder_name + "/models/" + network_name + "/"
            if not os.path.exists(model_folder_path):
                invalid = False
            else:
                network_name = input("\nThere is a network with that name already.\
                                        \nPlease choose a new name: ")

        model = choose_model(game)

        alpha_config_path = "Configs/Config_files/local_train_config.ini"
        search_config_path = "Configs/Config_files/local_search_config.ini"
        print("\nThe default config paths are:\n " + alpha_config_path + "\n " + search_config_path)

        use_default_configs = input("\nDo you wish to use these configs?(y/n)")
        if use_default_configs == "n":
            print("\nYou will new to provide new configs.")
            alpha_config_path = input("\nAlpha config path: ")
            search_config_path = input("\nSearch config path: ")

        alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
        alpha_zero.run()

def choose_game():
    available_games = ("SCS", "tic_tac_toe")

    game_question = "\nWhat game to you wish to play?\
                         \nType one of the following: "
    for g in available_games:
        game_question += "\n-> " + g

    game_question += "\n\nGame choice: "
    game_to_play = input(game_question)

    match game_to_play:
        case "SCS":
            game_class = SCS_Game
            print("\nUsing randomized configuration for the SCS game.")
            game_args = ["SCS/Game_configs/randomized_config.yml"]
        case "tic_tac_toe":
            game_class = tic_tac_toe
            game_args = []
        case _:
            print("Game unsupported in interative mode.")
            exit()

    return game_class, game_args

def choose_model(game):
    available_models = ("MLP", "ConvNet", "ResNet", "Recurrent")

    model_question = "\nWhat model to you wish to train?\
                         \nType one of the following: "
    for g in available_models:
        model_question += "\n-> " + g

    model_question += "\n\nModel choice: "
    model_to_use = input(model_question)

    hex_answer = input("\n\nWill the model use hexagonal convolutions?(y/n)")
    if hex_answer == "y":
        hexagonal = True
    else:
        hexagonal = False

    print("\nA model will be created based on the selected game.")

    match model_to_use:
        case "MLP":
            num_actions = game.get_num_actions()
            model = MLP_Net(num_actions)

        case "ConvNet":
            in_channels = game.get_state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            num_filters = input("\nNumber of filters: ")  
            kernel_size = input("Kernel size (int): ")  

            if hexagonal:
                model = Hex_ConvNet(in_channels, policy_channels, int(kernel_size), int(num_filters))
            else:
                model = Ort_ConvNet(in_channels, policy_channels, int(kernel_size), int(num_filters))

        case "ResNet":
            in_channels = game.get_state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            num_blocks = input("\nNumber of residual blocks: ")
            num_filters = input("Number of filters: ")  
            kernel_size = input("Kernel size (int): ")  

            if hexagonal:
                model = Hex_ResNet(in_channels, policy_channels, int(num_blocks), int(kernel_size), int(num_filters))
            else:
                model = Ort_ResNet(in_channels, policy_channels, int(num_blocks), int(kernel_size), int(num_filters))

        case "Recurrent":
            in_channels = game.get_state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            filters = input("\nNumber of filters to use internally:")      

            if hexagonal:
                model = Hex_RecurrentNet(in_channels, policy_channels, int(filters))
            else:
                model = Ort_DTNet(in_channels, policy_channels, int(filters))
                

        case _:
            print("Model type unsupported in interative mode.")
            exit()

    return model

def choose_method():
    method_answer = input("\nTest using mcts, raw policy or random agent?\
                               \n1 -> MCTS\
                               \n2 -> Policy\
                               \n3 -> Random\
                               \n\nNumber: ")
    if method_answer == "1":
        method = "mcts"
    elif method_answer == "2":
        method = "policy"
    elif method_answer == "3":
        method = "random"
    else:
        print("\nBad answer.")
        exit()

    return method

def choose_trained_network():
    network_name = input("\n\nName of the trained network: ")
    model_iteration_answer = input("\nModel iteration number: ")
    recurrent_answer = input("\n(This will be ignored if the network is not recurrent)\n" +
                                  "Number of recurrent iterations: ")
    model_iteration = int(model_iteration_answer)
    recurrent_iterations = int(recurrent_answer)
    return network_name, model_iteration, recurrent_iterations

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------    UTILITIES    -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def continue_training(game_class, game_args, trained_network_name, continue_network_name, use_same_configs, new_alpha_config_path=None, new_search_config_path=None, state_set=None):
    game = game_class(*game_args)

    game_folder_name = game.get_name()
    trained_model_folder_path = game_folder_name + "/models/" + trained_network_name + "/"
    if not os.path.exists(trained_model_folder_path):
        print("Could not find a model with that name.\n \
                If you are using Ray jobs with a working_directory,\
                only the models uploaded to git will be available.")
        exit()

    plot_data_load_path = trained_model_folder_path + "plot_data.pkl"

    pickle_path =  trained_model_folder_path + "base_model.pkl"
    with open(pickle_path, 'rb') as file:
        model = pickle.load(file)

    model_paths = glob.glob(trained_model_folder_path + "*_model")
            
    # finds all numbers in string -> gets the last one -> converts to int -> orders the numbers -> gets last number
    starting_iteration = sorted(list(map(lambda str: int(re.findall('\d+',  str)[-1]), model_paths)))[-1]
    latest_model_path = trained_model_folder_path + trained_network_name + "_" + str(starting_iteration) + "_model"
    model.load_state_dict(torch.load(latest_model_path, map_location=torch.device('cpu')))

    if use_same_configs:
        alpha_config_path = trained_model_folder_path + "train_config_copy.ini"
        search_config_path = trained_model_folder_path + "search_config_copy.ini"
    else:
        if new_search_config_path is None or new_alpha_config_path is None:
            print("If you are not using the same configs, you need to provide the new configs.")
            exit()
        alpha_config_path = new_alpha_config_path
        search_config_path = new_search_config_path

    alpha_zero = AlphaZero(game_class, game_args, model, continue_network_name, alpha_config_path, search_config_path, plot_data_path=plot_data_load_path, state_set=state_set)
    alpha_zero.run(starting_iteration)

def load_trained_network(game, net_name, model_iteration):

    game_folder = game.get_name() + "/"
    model_folder = game_folder + "models/" + net_name + "/" 
    pickle_path =  model_folder + "base_model.pkl"
    search_config_path = model_folder + "search_config_copy.ini"

    trained_model_path =  model_folder + net_name + "_" + str(model_iteration) + "_model"

    with open(pickle_path, 'rb') as file:
        model = pickle.load(file)

    model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu')))
    

    nn = Torch_NN(model)

    search_config = Search_Config()
    search_config.load(search_config_path)

    return nn, search_config

def test_loop(num_testers, method, num_games, game_class, game_args, AI_player=None, search_config=None, nn=None, recurrent_iterations=None, use_state_cache=None):

    wins = [0,0]
    
    if method == "random":
        args_list = [None]
        game_index = 0
    elif method == "policy":
        args_list = [AI_player, None, nn, None, recurrent_iterations, False]
        game_index = 1
    elif method == "mcts":
        args_list = [AI_player, search_config, None, nn, None, recurrent_iterations, False, False]
        game_index = 2
    elif method == "agent":
        args_list = [AI_player, None, False]
        game_index = 1

    actor_list = [RemoteTester.remote(print=False) for a in range(num_testers)]
    actor_pool = ray.util.ActorPool(actor_list)

    text = "Testing using " + method
    bar = PrintBar(text, num_games, 15)

    # We must use map instead of submit,
    # because ray bugs if you do several submit calls with different values
    map_args = []
    for g in range(num_games):
        args_copy = copy.copy(args_list)
        args_copy[game_index] = game_class(*game_args)
        map_args.append(args_copy)


    if method == "random":
        results = actor_pool.map_unordered(lambda actor, args: actor.random_vs_random.remote(*args), map_args)
    elif method == "policy":
        results = actor_pool.map_unordered(lambda actor, args: actor.Test_AI_with_policy.remote(*args), map_args)
    elif method == "mcts":
        results = actor_pool.map_unordered(lambda actor, args: actor.Test_AI_with_mcts.remote(*args), map_args)
    elif method == "agent":
        results = actor_pool.map_unordered(lambda actor, args: actor.Test_agent_vs_random.remote(*args), map_args)
        
    time.sleep(2)

    for res in results:
        winner, _ = res
        if winner != 0:
            wins[winner-1] +=1
        bar.next()
			
    bar.finish()

    # STATISTICS
    cmp_winrate_1 = 0.0
    cmp_winrate_2 = 0.0
    draws = num_games - wins[0] - wins[1]
    p1_winrate = wins[0]/num_games
    p2_winrate = wins[1]/num_games
    draw_percentage = draws/num_games
    cmp_2_string = "inf"
    cmp_1_string = "inf"

    if wins[0] > 0:
        cmp_winrate_2 = wins[1]/wins[0]
        cmp_2_string = format(cmp_winrate_2, '.4')
    if wins[1] > 0:  
        cmp_winrate_1 = wins[0]/wins[1]
        cmp_1_string = format(cmp_winrate_1, '.4')

  
    print("P1 Win ratio: " + format(p1_winrate, '.4'))
    print("P2 Win ratio: " + format(p2_winrate, '.4'))
    print("Draw percentage: " + format(draw_percentage, '.4'))
    print("Comparative Win ratio(p1/p2): " + cmp_1_string)
    print("Comparative Win ratio(p2/p1): " + cmp_2_string + "\n", flush=True)

    return p1_winrate, p2_winrate, draw_percentage

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
   
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

if __name__ == "__main__":
    main()