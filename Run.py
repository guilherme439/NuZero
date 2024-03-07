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

                alpha_config_path="Configs/Training/small_test_training_config.yaml"
                search_config_path="Configs/Search/test_search_config.yaml"

                state_set = create_r_unbalanced_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=12, policy_head="conv", value_head="reduce", hex=True)

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param)
                    
                #'''

                print("\n")
                context = start_ray_local(log_to_driver)
                alpha_zero = AlphaZero(game_class, game_args_list, alpha_config_path, search_config_path, model=model, state_set=state_set)
                alpha_zero.run()


            case 3:
                game_class = SCS_Game
                game_args_list = [["Games/SCS/Game_configs/r_unbalanced_config_5.yml"]]
                game = game_class(*game_args_list[0])

                alpha_config_path="Configs/Config_Files/Training/small_test_training_config.yaml"
                search_config_path="Configs/Config_Files/Search/test_search_config.yaml"


                state_set = create_r_unbalanced_state_set(game)

                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = RecurrentNet(in_channels, policy_channels, 256, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="tanh", hex=True)
                #model = ResNet(in_channels, policy_channels, num_filters=256, num_blocks=12, policy_head="conv", value_head="reduce", hex=True)

                #'''
                for name, param in model.named_parameters():
                    if ".weight" not in name:
                        #torch.nn.init.uniform_(param, a=-0.04, b=0.04)
                        torch.nn.init.xavier_uniform_(param)
                    
                #'''

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
                
                print("\n\nNeeds to be updated. Currently not working...\n")

            case 1: # Render Game
                rendering_mode = "interactive"  # passive | interactive

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/r_unbalanced_config_5.yml"]
                game = game_class(*game_args)

                # network options
                net_name = "explor_unbalanced_c3"
                model_iteration = 2040
                recurrent_iterations = 6


                nn, search_config = load_trained_network(game, net_name, model_iteration)

                # cache
                cache_choice = "keyless"
                max_size = 5000
                keep_updated = False

                
                # Agents
                mcts_agent = MctsAgent(search_config, nn, recurrent_iterations)
                policy_agent = PolicyAgent(nn, recurrent_iterations)
                random_agent = RandomAgent()
                goal_agent = GoalRushAgent()
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
                num_testers = 4
                num_games = 100

                game_class = SCS_Game
                game_config = "SCS/Game_configs/solo_soldier_config_5.yml"
                game_args = [game_config]
                game = game_class(*game_args)


                # cache
                cache_choice = "keyless"
                max_size = 5000
                keep_updated = False

                cache = create_cache(cache_choice, max_size)
                
                ## trained network
                net_name = "mirror_plus_cl"
                model_iteration = 800
                recurrent_iterations = 6

                # search config
                search_config_path = "Configs/Config_Files/Search/small_test_search_config.ini"

                new_net = True
                if new_net:
                    in_channels = game.get_state_shape()[0]
                    policy_channels = game.get_action_space_shape()[0]
                    model = RecurrentNet(in_channels, policy_channels, 128, 2, recall=True, policy_head="conv", value_head="reduce", value_activation="relu", hex=True)
                    nn = Torch_NN(model)
                else:
                    nn, search_config = load_trained_network(game, net_name, model_iteration)

                load_config = True
                if load_config:
                    search_config = Search_Config()
                    search_config.load(search_config_path)


                # test manager
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, keep_updated)

                
                # Agents
                mcts_agent = MctsAgent(search_config, nn, recurrent_iterations, cache)
                policy_agent = PolicyAgent(nn, recurrent_iterations, cache)
                random_agent = RandomAgent()
                goal_agent = GoalRushAgent()
                p1_agent = mcts_agent
                p2_agent = mcts_agent

                ################################################
                print("\n")
                print(net_name)
                print(game_args)
                print("Testing with " + str(recurrent_iterations) + " recurrent iterations.\n")

                test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)

                time.sleep(1)

            case 3: # Graphs for several network checkpoints
                ray.init()

                num_testers = 5
                num_games = 200

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config_5.yml"]
                game = game_class(*game_args)

                recurrent_iterations = 10

                # network options
                net_name = "mirror_final_atempt"


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
                start_ray_local(log_to_driver)

                num_testers = 3
                num_games = 200

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config_20.yml"]
                game = game_class(*game_args)


                # network options
                net_name = "new_solo_2_c"
                model_iteration = 2220

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                

                #---
                min = 0
                max = 30
                step = 1
                recurrent_iterations_list = range(min,max+1,step)
                
                name = net_name + "_" + str(model_iteration) + "_15x15_" + str(min) + "-" + str(max) + "-iterations"
                figpath = "Graphs/iterations/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "keyless",1000)
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

                print("done!")
            

            case 5: # Graphs for several games (can be used to compared performance with board size for example)
                ray.init()

                game = SCS_Game("SCS/Game_configs/solo_soldier_config_5.yml")

                num_testers = 5
                num_games = 100

                # network options
                net_name = "solo_res"
                model_iteration = 1100
                rec_iter = recurrent_iterations = 1

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
                                "SCS/Game_configs/solo_soldier_config_10.yml",]
                            
                
                
                name = net_name + "_" + str(model_iteration) + "_5x5_to_10x10_" + str(recurrent_iterations) + "-iterations_mcts"
                figpath = "Graphs/sizes/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "keyless",1000)
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()

                

                p1_wr_list = []
                p2_wr_list = []
                for config in configs_list:
                    game_args = [config]
                    print("\n" + str(game_args))
                    test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                    p1_agent = RandomAgent()
                    p2_agent = MctsAgent(search_config, nn, rec_iter, "dict")
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(range(len(configs_list)), p1_wr_list, label = "P1")
                plt.plot(range(len(configs_list)), p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

                name = "solo_res_5x5_to_10x10_mcts_extrapolation"
                data = p2_wr_list
                save_path = "Graphs/_graph_data/" + name + '.pkl'
                pickle_save(save_path, data)

            case 6: # Graphs for sequence of games and iterations
                ray.init()

                game = SCS_Game("SCS/Game_configs/solo_soldier_config_5.yml")

                num_testers = 3
                num_games = 250

                # network options
                net_name = "new_solo_2_c"
                model_iteration = 2220

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
                                "SCS/Game_configs/solo_soldier_config_15.yml",
                                "SCS/Game_configs/solo_soldier_config_16.yml",
                                "SCS/Game_configs/solo_soldier_config_17.yml",
                                "SCS/Game_configs/solo_soldier_config_18.yml",
                                "SCS/Game_configs/solo_soldier_config_19.yml",
                                "SCS/Game_configs/solo_soldier_config_20.yml"]
                            
                #---
                min = 3
                max = 19
                step = 1
                recurrent_iterations_list = list(range(min,max+1,step))
                
                name = str(min) + "-" + str(max) + "_increasing_size_" + net_name + "_" + str(model_iteration)
                figpath = "Graphs/comb/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "per_game")
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()

                start = 4

                p1_wr_list = []
                p2_wr_list = []
                for i in range(len(configs_list)):
                    print("\ni: " + str(start+i))
                    game_args = [configs_list[i]]
                    rec_iter = recurrent_iterations_list[i]
                    test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                    p1_agent = RandomAgent()
                    p2_agent = PolicyAgent(nn, rec_iter)
                    p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                    p1_wr_list.append(p1_wr)
                    p2_wr_list.append(p2_wr)


                plt.plot(recurrent_iterations_list, p1_wr_list, label = "P1")
                plt.plot(recurrent_iterations_list, p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

                print("done!")

            case 7: # extrapolation testing with multiple runs
                start_ray_local(log_to_driver)

                num_testers = 3
                num_runs = 4
                num_games = 100

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/solo_soldier_config_7.yml"]
                game = game_class(*game_args)


                # network options
                net_name = "solo_final"
                model_iteration = 1100

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn)
                test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                

                #---
                min = 0
                max = 100
                step = 1
                recurrent_iterations_list = list(range(min,max+1,step))
                
                name = net_name + "_" + str(model_iteration) + "_7x7_" + str(min) + "-" + str(max) + "-iterations"
                figpath = "Graphs/iterations/" + name
                print(figpath)

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "keyless",1000)
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent(game)

                num_rec_iters = len(recurrent_iterations_list)
                p1_wr_list = [0] * num_rec_iters
                p2_wr_list = [0] * num_rec_iters
                
                for run in range(num_runs):
                    for i in range(num_rec_iters):
                        rec_iter = recurrent_iterations_list[i]
                        p1_agent = RandomAgent()
                        p2_agent = PolicyAgent(nn, rec_iter)
                        print("\n\n\nTesting with " + str(rec_iter) + " iterations\n")
                        p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                        p1_wr_list[i] += p1_wr/num_runs
                        p2_wr_list[i] += p2_wr/num_runs


                plt.plot(recurrent_iterations_list, p1_wr_list, label = "P1")
                plt.plot(recurrent_iterations_list, p2_wr_list, label = "P2")
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

                print("done!")

            case 8: # Multiple extrapolation runs with different maps
                start_ray_local(log_to_driver)

                num_testers = 5
                num_runs_per_game = 2
                num_games = 100

                game_class = SCS_Game
                configs_list = ["SCS/Game_configs/solo_soldier_config_5.yml",
                                "SCS/Game_configs/solo_soldier_config_6.yml",
                                "SCS/Game_configs/solo_soldier_config_7.yml",
                                "SCS/Game_configs/solo_soldier_config_8.yml",
                                "SCS/Game_configs/solo_soldier_config_9.yml",
                                "SCS/Game_configs/solo_soldier_config_10.yml"]
                
                game = SCS_Game("SCS/Game_configs/solo_soldier_config_5.yml")

                # network options
                net_name = "solo_reduce_prog_4"
                model_iteration = 1100            

                # iteration options
                min = 0
                max = 100
                step = 1
                recurrent_iterations_list = list(range(min,max+1,step))
                
                name = net_name + "_" + str(model_iteration) + "_" + str(min) + "-" + str(max) + "-iterations_extrapolation"
                figpath = "Graphs/" + name
                print(figpath)

                

                # Test Manager configuration
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                shared_storage = RemoteStorage.remote(window_size=1)
                shared_storage.store.remote(nn) 

                ################################################

                #mcts_agent = MctsAgent(search_config, nn, rec_iter, "keyless",1000)
                #policy_agent = PolicyAgent(nn, rec_iter)
                #random_agent = RandomAgent()
                #goal_agent = GoalRushAgent()

                num_rec_iters = len(recurrent_iterations_list)
                #p1_wr_list = [([0] * num_rec_iters) for game in range(len(configs_list))]
                p2_wr_list = [([0] * num_rec_iters) for game in range(len(configs_list))]

                #print(p1_wr_list)
                #print(p2_wr_list)
                
                for i in range(len(configs_list)):
                    print("game_idx: " + str(i))
                    game_args = [configs_list[i]]
                    test_manager = TestManager(game_class, game_args, num_testers, shared_storage, None)
                    for run in range(num_runs_per_game):
                        print("run " + str(run))
                        for k in range(num_rec_iters):
                            rec_iter = recurrent_iterations_list[k]
                            p1_agent = RandomAgent()
                            p2_agent = PolicyAgent(nn, rec_iter)
                            print("\n\n\nTesting with " + str(rec_iter) + " iterations\n")
                            p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                            #p1_wr_list[i][k] += p1_wr/num_runs_per_game
                            p2_wr_list[i][k] += p2_wr/num_runs_per_game

                    #plt.plot(recurrent_iterations_list, p1_wr_list, label = str(i+5))
                    plt.plot(recurrent_iterations_list, p2_wr_list[i], label = str(i+5) + "x" + str(i+5))
                    plt.savefig(figpath)

                
                plt.title(name)
                plt.legend()
                plt.savefig(figpath)
                plt.clf()

                data = p2_wr_list
                save_path = "Graphs/_graph_data/" + name + '.pkl'
                pickle_save(save_path, data)

                print("done!")

            case 9: # build graphs from data
                
                figname = "30_vs_res"
                titulo = "Architecture Comparison"
                plt.figure(figsize=(12, 6.8))

                data_path = "Graphs/_graph_data/solo_reduce_prog_4_1100_0-100-iterations_extrapolation.pkl"
                prog_win_rates = pickle_load(data_path)

                data_path = "Graphs/_graph_data/solo_res_5x5_to_10x10_win_rates.pkl"
                res_wr = pickle_load(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
                    color = line_lst[0].get_color()
                    wr = res_wr[size_i]
                    plt.axhline(y=wr, linestyle='--', label = str(size_i+5) + "x" + str(size_i+5) + "_ResNet", color = color)

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.1 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return
            
            case 10: # build more graphs from data
                
                figname = "solo_extrapolation"
                titulo = "Solo Soldier Extrapolation"
                plt.figure(figsize=(12, 7))

                data_path = "Graphs/_graph_data/solo_final_1100_0-100-iterations.pkl"
                prog_win_rates = pickle_load(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
        

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.05 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return
            
            case 11: # build even more graphs from data
                
                figname = "30_vs_res"
                titulo = "Architecture Comparison"
                plt.figure(figsize=(12, 6.8))

                data_path = "Graphs/_graph_data/solo_reduce_prog_4_1100_0-100-iterations_extrapolation.pkl"
                prog_win_rates = pickle_load(data_path)

                data_path = "Graphs/_graph_data/solo_res_5x5_to_10x10_win_rates.pkl"
                res_wr = pickle_load(data_path)

                for size_i in range(len(prog_win_rates)):
                    iterations = prog_win_rates[size_i]
                    line_lst = plt.plot(range(len(iterations)), iterations, label = str(size_i+5) + "x" + str(size_i+5))
                    color = line_lst[0].get_color()
                    wr = res_wr[size_i]
                    plt.axhline(y=wr, linestyle='--', label = str(size_i+5) + "x" + str(size_i+5) + "_ResNet", color = color)

                plt.xlabel("Recurrent Iterations")
                plt.ylabel("Win Ratio")
        
                plt.title(titulo, pad=20, fontsize = 14)

                lgd = plt.legend(bbox_to_anchor=(1,1))
                plt.gcf().canvas.draw()
                invFigure = plt.gcf().transFigure.inverted()

                lgd_pos = lgd.get_window_extent()
                lgd_coord = invFigure.transform(lgd_pos)
                lgd_xmax = lgd_coord[1, 0]

                ax_pos = plt.gca().get_window_extent()
                ax_coord = invFigure.transform(ax_pos)
                ax_xmax = ax_coord[1, 0]        

                shift = 1.1 - (lgd_xmax - ax_xmax)
                plt.gcf().tight_layout(rect=(0, 0, shift, 1))


                plt.savefig("Graphs/" + figname, dpi=300)
                plt.clf()

                return


            case _:
                print("Unknown testing preset.")
                return

        
        
            
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

            case 1: # Create unit images manually
                
                renderer = SCS_Renderer()

                renderer.create_marker_from_scratch("ally_soldier", (1,2,2), "infantary", color_str="dark_green")
                renderer.add_border("green", "SCS/Images/ally_soldier.jpg")

                renderer.create_marker_from_scratch("ally_tank", (2,2,4), "mechanized", color_str="dark_green")
                renderer.add_border("green", "SCS/Images/ally_tank.jpg")

                renderer.create_marker_from_scratch("axis_soldier", (1,1,3), "infantary", color_str="dark_red")
                renderer.add_border("red", "SCS/Images/axis_soldier.jpg")

                renderer.create_marker_from_scratch("axis_tank", (4,6,1), "mechanized", color_str="dark_red")
                renderer.add_border("red", "SCS/Images/axis_tank.jpg")


                print("\nImages created!\n")

            case 2:
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