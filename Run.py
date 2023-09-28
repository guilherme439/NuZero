import sys
import os
import psutil
import time
import random
import pickle
import ray
import copy
import argparse
import glob
import re

import torch
import numpy

from progress.bar import ChargingBar
from PrintBar import PrintBar

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network as MLP_Net

from Neural_Networks.Orthogonal.Simple_Conv_Network import Simple_Conv_Network as Ort_ConvNet
from Neural_Networks.Orthogonal.ResNet import ResNet as Ort_ResNet
from Neural_Networks.Orthogonal.dt_neural_network import dt_net_2d as Ort_DTNet
from Neural_Networks.Orthogonal.dt_neural_network import dt_net_recall_2d as Ort_DTNet_recall

from Neural_Networks.Hexagonal.Simple_Conv_Network import Simple_Conv_Network as Hex_ConvNet
from Neural_Networks.Hexagonal.ResNet import ResNet as Hex_ResNet
from Neural_Networks.Hexagonal.dt_neural_network import dt_net_2d as Hex_DTNet
from Neural_Networks.Hexagonal.dt_neural_network import dt_net_recall_2d as Hex_DTNet_recall


from SCS.SCS_Game import SCS_Game
from SCS.SCS_Renderer import SCS_Renderer

from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Configs.AlphaZero_config import AlphaZero_config
from Configs.Search_config import Search_config

from AlphaZero import AlphaZero
from Tester import Tester
from RemoteTester import RemoteTester

from stats_utilities import *

from Gamer import Gamer
from Replay_Buffer import Replay_Buffer
from Shared_network_storage import Shared_network_storage

from ray.runtime_env import RuntimeEnv

'''
~/Desktop/ray_tmp/session_latest/runtime_resources/working_dir_files

python SLURM/slurm-launch.py --exp-name two_nodes_gaips --num-nodes 2 --gaips --node nexus[2-3] --net-name tests_on_gaips

srun -w nexus4 ray stop
--------------------------------------------

ray job submit --address="http://127.0.0.1:8265" --runtime-env-json='{"working_dir": "https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip", "pip": "./requirements.txt"}' -- python Run.py --training-preset 3

python Run.py --training-preset 1 --name good_name

'''


def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    parser = argparse.ArgumentParser()
    exclusive_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument(
        "--name", type=str, default="",
        help="Change the name of network trained with the preset")
    exclusive_group.add_argument(
        "--interactive", action='store_true',
        help="Create a simple training setup interactivly"
    )
    exclusive_group.add_argument(
        "--training-preset", type=int,
        help="Choose one of the preset training setups"
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
               
    if args.training_preset is not None:
        match args.training_preset:
            case 0: # Tic_tac_toe example
                game_class = tic_tac_toe
                game_args = []
                game = game_class(*game_args)

                alpha_config_path="Configs/Config_files/ttt_alpha_config.ini"
                search_config_path="Configs/Config_files/ttt_search_config.ini"

                network_name = "ttt_net"

                ################################################

                if args.name is not None and args.name != "":
                    network_name = args.name

                num_actions = game.get_num_actions()
                model = MLP_Net(num_actions)

                print("\n")
                context = start_ray_local()
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
                alpha_zero.run()

            case 1: # Run on local machine
                
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config_super_soldiers.yml"]
                game = game_class(*game_args)

                alpha_config_path="Configs/Config_files/slice_alpha_config.ini"
                search_config_path="Configs/Config_files/slice_search_config.ini"

                network_name = "local_net"

                ################################################

                in_channels = game.state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_DTNet(in_channels, policy_channels, 350)

                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local()
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
                alpha_zero.run()

            case 2: # Run on local machine within a cluster
                    # Local machine runs on local files and remote nodes use git repository

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config.yml"]
                game = game_class(*game_args)

    
                alpha_config_path="Configs/Config_files/SCS_alpha_config.ini"
                search_config_path="Configs/Config_files/SCS_search_config.ini"

                network_name = "slice_test"

                ################################################

                in_channels = game.state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_DTNet(in_channels, policy_channels, 256)
                
                if args.name is not None and args.name != "":
                    network_name = args.name

                print("\n")
                context = start_ray_local_cluster()
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
                alpha_zero.run()

            case 3: # Run on remote cluster using ray jobs API
                # The runtime environment is specified when launching the job
                
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config.yml"]
                game = game_class(*game_args)
                
                alpha_config_path="Configs/Config_files/SCS_alpha_config.ini"
                search_config_path="Configs/Config_files/SCS_search_config.ini"
                
                network_name = "cluster_net"

                ################################################

                in_channels = game.state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]
                model = Hex_DTNet(in_channels, policy_channels, 256)

                if args.name is not None and args.name != "":
                    network_name = args.name
                          
                alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
                alpha_zero.run()

            case 4: # Continue Training

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config.yml"]

                trained_network_name = "local_alphazero_continue"
                continue_network_name = "local_alphazero_continue_2" # new network can have the same name as the previous
                use_same_configs = True

                # In case of not using the same configs define the new configs to use like this
                new_alpha_config_path="Configs/Config_files/local_alpha_config.ini"
                new_search_config_path="Configs/Config_files/local_search_config.ini"

                ################################################

                print("\n")
                context = start_ray_local()
                continue_training(game_class, game_args, trained_network_name, continue_network_name, \
                                  use_same_configs, new_alpha_config_path, new_search_config_path)


            case 6:
                # Define your setup here
                exit()


    elif args.testing_preset is not None:
        match args.testing_preset:
            case 0: # Example
                pass

            case 1: # Render Game
                rendering_mode = "interactive"  # passive | interactive

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config.yml"]
                method = "mcts"

                # testing options
                AI_player = "both"
                recurrent_iterations = 3

                # network options
                net_name = "local_alphazero_continue_2"
                model_iteration = 100

                # TODO: Add possibilty of using second network

                ################################################

                game = game_class(*game_args)
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                
                if rendering_mode == "passive":
                    tester = Tester(render=True)
                elif rendering_mode == "interactive":
                    tester = Tester(print=True)

                if method == "mcts":
                    tester.Test_AI_with_mcts(AI_player, search_config, game, nn, use_state_cache=False, recurrent_iterations=recurrent_iterations)
                elif method == "policy":
                    tester.Test_AI_with_policy(AI_player, game, nn, recurrent_iterations=recurrent_iterations)
                elif method == "random":
                    tester.random_vs_random(game)

                print("\n\nLength: " + str(game.get_length()) + "\n")

                if rendering_mode == "interactive":
                    time.sleep(0.5)

                    renderer = SCS_Renderer.remote()
                    end = renderer.analyse.remote(game, True)

                    ray.get(end) # wait for the rendering to end

            case 2: # Statistics for Multiple Games
                ray.init()

                number_of_testers = 5

                game_class = SCS_Game
                game_args = ["SCS/Game_configs/mirrored_config_super_soldiers.yml"]
                method = "random"

                # testing options
                num_games = 100
                AI_player = "2"
                recurrent_iterations = 2

                # network options
                net_name = "local_alphazero_continue_2"
                model_iteration = 100

                # TODO: Add possibilty of using second network

                ################################################
                
                game = game_class(*game_args)
                nn, search_config = load_trained_network(game, net_name, model_iteration)
                
                test_loop(number_of_testers, method, num_games, game_class, game_args, AI_player, search_config, nn, recurrent_iterations, False)

            case _:
                print("Unknown testing preset.")
                return


    elif args.interactive:
        print("\nStarted interactive mode!\n")
        print("\nAnswer the questions to start training or testing.\n")

        game_class, game_args = choose_game()
        game = game_class(*game_args)
        game_folder_name = game.get_name()

        continue_answer = input("\nDo you wish to continue training a previous network or train a new one?(1 or 2)\
                                 \n1 -> Continue training\
                                 \n2 -> New Network\n\n")
        
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

            alpha_config_path = "Configs/Config_files/default_alpha_config.ini"
            search_config_path = "Configs/Config_files/default_search_config.ini"
            print("\nThe default config paths are:\n " + alpha_config_path + "\n " + search_config_path)

            change_configs = input("\nDo you wish to use these configs?(y/n)")
            if change_configs == "y":
                print("\nYou will new to provide new configs.")
                alpha_config_path = input("\nAlpha config path: ")
                search_config_path = input("\nSearch config path: ")

            alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
            alpha_zero.run()


    elif args.debug is not None:
        match args.debug:
            case 0:
                game_class = SCS_Game
                game_args = ["SCS/Game_configs/randomized_config.yml"]
                game = game_class(*game_args)

                tester = Tester(print=False)

                #tester.random_vs_random(game, keep_state_history=True)
                #random_index = np.random.choice(range(len(game.state_history)))
                #state_image = game.state_history[random_index]
                #game.debug_state_image(state_image)

                #play_loop(10000, game_class, game_args, tester)

            case 1: # default search config
                search_config = Search_config()

                filepath = "Configs/Config_files/default_search_config.ini"
                search_config.save(filepath)

            case 2: # default alpha config
                alpha_config = AlphaZero_config()

                filepath = "Configs/Config_files/default_alphazero_config.ini"
                alpha_config.save(filepath)

            case 3:
                renderer = SCS_Renderer.remote()
                name = "floribela"
                lindura = renderer.create_unit_image.remote(name, 1, (2,5,4))
                ray.get(lindura)
                print(name)
    
    return

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------   INTERACTIVE   -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

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
            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            num_filters = input("\nNumber of filters: ")  
            kernel_size = input("\nKernel size (int): ")  

            if hexagonal:
                model = Hex_ConvNet(in_channels, policy_channels, int(kernel_size), int(num_filters))
            else:
                model = Ort_ConvNet(in_channels, policy_channels, int(kernel_size), int(num_filters))

        case "ResNet":
            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            num_blocks = input("\nNumber of residual blocks: ")
            num_filters = input("\nNumber of filters: ")  
            kernel_size = input("\nKernel size (int): ")  

            if hexagonal:
                model = Hex_ResNet(in_channels, policy_channels, int(num_blocks), int(kernel_size), int(num_filters))
            else:
                model = Ort_ResNet(in_channels, policy_channels, int(num_blocks), int(kernel_size), int(num_filters))

        case "Recurrent":
            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            filters = input("\nNumber of filters to use internally:")      

            if hexagonal:
                model = Hex_DTNet(in_channels, policy_channels, int(filters))
            else:
                model = Ort_DTNet(in_channels, policy_channels, int(filters))
                

        case _:
            print("Model type unsupported in interative mode.")
            exit()

    return model

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------    UTILITIES    -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def continue_training(game_class, game_args, trained_network_name, continue_network_name, use_same_configs, new_alpha_config_path=None, new_search_config_path=None):
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
    model.load_state_dict(torch.load(latest_model_path))

    if use_same_configs:
        alpha_config_path = trained_model_folder_path + "alpha_config_copy.ini"
        search_config_path = trained_model_folder_path + "search_config_copy.ini"
    else:
        if new_search_config_path is None or new_alpha_config_path is None:
            print("If you are not using the same configs, you need to provide the new configs.")
            exit()
        alpha_config_path = new_alpha_config_path
        search_config_path = new_search_config_path

    alpha_zero = AlphaZero(game_class, game_args, model, continue_network_name, alpha_config_path, search_config_path, plot_data_path=plot_data_load_path)
    alpha_zero.run(starting_iteration)

def load_trained_network(game, net_name, model_iteration):

    game_folder = game.get_name() + "/"
    model_folder = game_folder + "models/" + net_name + "/" 
    pickle_path =  model_folder + "base_model.pkl"
    search_config_path = model_folder + "search_config_copy.ini"

    trained_model_path =  model_folder + net_name + "_" + str(model_iteration) + "_model"

    with open(pickle_path, 'rb') as file:
        model = pickle.load(file)
    model.load_state_dict(torch.load(trained_model_path))

    nn = Torch_NN(game, model)

    search_config = Search_config()
    search_config.load(search_config_path)

    return nn, search_config

def test_loop(num_testers, method, num_games, game_class, game_args, AI_player=None, search_config=None, nn=None, recurrent_iterations=None, use_state_cache=None):

    wins = [0,0]
    
    if method == "random":
        args_list = [None]
        game_index = 0
    elif method == "policy":
        args_list = [AI_player, None, nn, recurrent_iterations]
        game_index = 1
    elif method == "mcts":
        args_list = [AI_player, search_config, None, nn, None, recurrent_iterations, False, False]
        game_index = 2

    
    actor_list = [RemoteTester.remote() for a in range(num_testers)]
    actor_pool = ray.util.ActorPool(actor_list)

    text = "Testing using " + method
    bar = PrintBar(text, num_games, 15)

    for g in range(num_games):
        game = game_class(*game_args)
        args_list[game_index] = game
        if method == "random":
            actor_pool.submit(lambda actor, args: actor.random_vs_random.remote(*args), args_list)
        elif method == "policy":
            actor_pool.submit(lambda actor, args: actor.Test_AI_with_policy.remote(*args), args_list)
        elif method == "mcts":
            actor_pool.submit(lambda actor, args: actor.Test_AI_with_mcts.remote(*args), args_list)

    
    for g in range(num_games):
        winner, _ = actor_pool.get_next_unordered(250, True) # Timeout and Ignore_if_timeout
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

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
   
##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------       RAY       -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################

def start_ray_local():
    print("\n\n--------------------------------\n\n")
		
    context = ray.init(log_to_driver=False)
    return context

def start_ray_local_cluster():
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					pip="./requirements.txt"
					)
		
    context = ray.init(address='auto', runtime_env=runtime_env, log_to_driver=False)
    return context

def start_ray_slice():
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					conda="tese",
					)
		
    context = ray.init(runtime_env=runtime_env, log_to_driver=True)
    return context

if __name__ == "__main__":
    main()