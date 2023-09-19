import sys
import os, psutil
import gc
import time
import random
import pickle
import ray
import copy

import numpy as np

import torch
from torch import nn

from progress.bar import ChargingBar

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network

#from Neural_Networks.Rectangular.Simple_Conv_Network import Simple_Conv_Network
#from Neural_Networks.Rectangular.ResNet import ResNet
#from Neural_Networks.Rectangular.dt_neural_network import *

from Neural_Networks.Hexagonal.Simple_Conv_Network import Simple_Conv_Network
from Neural_Networks.Hexagonal.ResNet import ResNet
from Neural_Networks.Hexagonal.dt_neural_network import *


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
ray job submit --address="http://127.0.0.1:8265" --runtime-env-json='{"working_dir": "https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip", "pip": "./requirements.txt"}' -- python Run.py 0
'''

def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    #multiprocessing.set_start_method("spawn")

    if (len(sys.argv)!=2):
        print("Run with single mode argument.")
        return
    else:
        mode = int(sys.argv[1])

    match mode:

        case 0: # Local
            #context = start_ray_local()
            
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/detailed_config.yml"]
            game = game_class(*game_args)

            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            model = dt_net_2d(in_channels, policy_channels, 256)
            #model = ResNet(in_channels, policy_channels, num_blocks=3, kernel_size=1, num_filters=128)

            alpha_zero = AlphaZero(game_class, game_args, model=model, 
                                   default_alpha_config="Configs/Config_files/test_alpha_config.ini", 
                                   default_search_config="Configs/Config_files/SCS_search_config.ini")
            alpha_zero.run()

        case 1: # Slice
            #context = start_ray_slice()
            
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/mirrored_config.yml"]
            game = game_class(*game_args)

            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            model = dt_net_2d(in_channels, policy_channels, 256)
            #model = ResNet(in_channels, policy_channels, num_blocks=3, kernel_size=1, num_filters=128)
            
            alpha_zero = AlphaZero(game_class, game_args, model=model, 
                                   default_alpha_config="Configs/Config_files/SCS_alpha_config.ini", 
                                   default_search_config="Configs/Config_files/SCS_search_config.ini")
            alpha_zero.run()

        case 2: # RNL
            
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/detailed_config.yml"]
            game = game_class(*game_args)

            in_channels = game.state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            model = dt_net_2d(in_channels, policy_channels, 256)
            #model = ResNet(in_channels, policy_channels, num_blocks=3, kernel_size=1, num_filters=128)            

            alpha_zero = AlphaZero(game_class, game_args, model=model, 
                                   default_alpha_config="Configs/Config_files/rnl_alpha_config.ini", 
                                   default_search_config="Configs/Config_files/SCS_search_config.ini")
            alpha_zero.run()

        case 3: # search config
            search_config = Search_config()

            filepath = "Configs/Config_files/default_search_config.ini"
            search_config.save(filepath)

        case 4: # alpha config
            alpha_config = AlphaZero_config()

            filepath = "Configs/Config_files/default_alphazero_config.ini"
            alpha_config.save(filepath)
            
        case 5: # test network
            
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/randomized_config.yml"]
            game = game_class(*game_args)

            test_trained_network(game)
            
        case 6: # Watch trained game
            
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/randomized_config.yml"]
            game = game_class(*game_args)

            test_trained_network(game)
            time.sleep(0.5)

            renderer = SCS_Renderer.remote()
            end = renderer.analyse.remote(game, True)

            ray.get(end) # wait for the rendering to end
            
        case 7: # Watch Random Game
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/detailed_config.yml"]
            game = game_class(*game_args)
            
            features = game.action_space_shape[0] * game.action_space_shape[1] * game.action_space_shape[2]
            model = MLP_Network(features)
            nn = Torch_NN(game, model, recurrent=False)

            search_config = Search_config()
            search_config.load("Configs/Config_files/test_search_config.ini")

            tester = Tester(print=True)
            #tester.set_slow_duration(5)
            #tester.Test_AI_with_mcts("1", search_config, game, nn, use_state_cache=False, recurrent_iterations=2)
            tester.random_vs_random(game)
            
            time.sleep(0.5)

            renderer = SCS_Renderer.remote()
            end = renderer.analyse.remote(game, True)

            ray.get(end) # wait for the rendering to end
            return

        case 8:
            print("\n\n GAMER! \n")
            start_ray()

            game_class = SCS_Game().__class__
            game_args = ["SCS/Game_configs/randomized_config.yml"]
            game = game_class(*game_args)

            model = dt_net_2d(game, 128)
            nn = Torch_NN(model, recurrent=True)

            search_config = Search_config()
            search_config.load("Configs/Config_files/test_search_config.ini")

            buffer = Replay_Buffer.remote(5000, 64)
            network_storage = Shared_network_storage.remote(4)
            network_storage.save_network.remote(nn)

            gamer = Gamer.remote(buffer, network_storage, game_class, game_args, search_config, 2, "disabled")
        
            stats = ray.get(gamer.play_game.remote())

            print_stats(stats)
            
        case 9:
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/randomized_config.yml"]
            game = game_class(*game_args)

            tester = Tester(print=False)

            #tester.random_vs_random(game, keep_state_history=True)
            #random_index = np.random.choice(range(len(game.state_history)))
            #state_image = game.state_history[random_index]
            #game.debug_state_image(state_image)

            play_loop(10000, game_class, game_args, tester)
            
        case 10:
            # reset_env tests

            game_class = SCS_Game
            game_args = ["SCS/Game_configs/randomized_config.yml"]
            game = game_class(*game_args)
            
            features = game.action_space_shape[0] * game.action_space_shape[1] * game.action_space_shape[2]
            model = MLP_Network(features)

            nn = Torch_NN(game, model, recurrent=False)

            search_config = Search_config()
            search_config.load("Configs/Config_files/test_search_config.ini")

            tester = Tester(print=True)
            #tester.set_slow_duration(5)
            #tester.Test_AI_with_mcts("1", search_config, game, nn, use_state_cache=False, recurrent_iterations=2)
            tester.random_vs_random(game)

            history_copy = copy.deepcopy(game.action_history)
            game.reset_env()
            for action in history_copy:
                game.step_function(action)


        case 11:
            pass
        
        case 12:
            pass

        case 13:
            pass

        case 14:
            pass

        case 15:
            pass

        case 16:
            game_class = tic_tac_toe().__class__
            game_args = []
            
            pickle_path = "Tic_Tac_Toe/models/fast_conv/Network.pkl"
            model = pickle.load(pickle_path)

            trained_model_path = "Tic_Tac_Toe/models/fast_conv/fast_conv_300_model"
            model.load_state_dict(torch.load(trained_model_path))

            alpha_config = Alpha_Zero_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, game_class, game_args, alpha_config, network_name=net_name)
        
            Alpha_Zero.run()

        case 17:
            pass

        case 18:
            game_class = SCS_Game().__class__
            game_args = [[3,1],[1,2], True]
            game = game_class(*game_args)

            model = dt_net_2d(game, 128)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_SCS_config()

            net_name = "Tests"

            Alpha_Zero = AlphaZero(model, True, game_class, game_args, alpha_config, network_name=net_name)
            Alpha_Zero.run()

        case 19:
            game_class = SCS_Game().__class__
            game_args = [[1,0],[0,0], True]         
            game = game_class(*game_args)
            
            features = game.action_space_shape[0] * game.action_space_shape[1] * game.action_space_shape[2]
            model = MLP_Network(features)

            nn = Torch_NN(model, recurrent=False)

            search_config = Search_config()
            search_config.load("Configs/Config_files/test_search_config.ini")

            tester = Tester(render=True)
            tester.set_slow_duration(1.3)
            
            #tester.Test_AI_with_mcts("both", game, search_config, nn, use_state_cache=False, recurrent_iterations=2)
            tester.random_vs_random(game)

            print("\n\nLength: " + str(game.length) + "\n")

            renderer = SCS_Renderer.remote()
            end = renderer.analyse.remote(game)


            ray.get(end) # wait for the rendering to end

        case 20:
            game_class = SCS_Game
            game_args = ["SCS/Game_configs/randomized_config.yml"]

            play_loop(1000, game_class, game_args)
            
        case 21:
            pass

        case _:
            print("default")

    return

##########################################################################
# ----------------------------               --------------------------- #
# ---------------------------    UTILITIES    -------------------------- #
# ----------------------------               --------------------------- #
##########################################################################


def start_ray_slice():
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					conda="tese",
                    env_vars=
							{
							"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
							}
					)
		
    context = ray.init(runtime_env=runtime_env, log_to_driver=True)
    return context


def start_ray_rnl():
    print("\n\n--------------------------------\n\n")

    runtime_env=RuntimeEnv \
					(
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					pip="./requirements.txt"
					)
		
    context = ray.init(address="auto", runtime_env=runtime_env, log_to_driver=True)
    return context

def start_ray_local():
    print("\n\n--------------------------------\n\n")
		
    context = ray.init(address="auto", log_to_driver=True)
    return context

def test_trained_network(game):

    net_name = input("\nName of the network: ")
    recurrent = False
    rec = input("\nRecurrent?(y/n):")
    if rec == "y":
        recurrent = True
    model_iteration = int(input("\nModel iteration number: "))
    AI_player = input("\nChose the AI player (\"1\", \"2\", \"both\"): ")

    ####################

    game_folder = game.get_name() + "/"
    model_folder = game_folder + "models/" + net_name + "/" 
    pickle_path =  model_folder + "base_model.pkl"
    search_config_path = model_folder + "search_config_copy.ini"

    trained_model_path =  model_folder + net_name + "_" + str(model_iteration) + "_model"

    with open(pickle_path, 'rb') as file:
        model = pickle.load(file)
    model.load_state_dict(torch.load(trained_model_path))

    nn = Torch_NN(game, model, recurrent=recurrent)

    search_config = Search_config()
    search_config.load(search_config_path)

    tester = Tester(print=True)
    tester.Test_AI_with_mcts(AI_player, search_config, game, nn, use_state_cache=True, recurrent_iterations=2)
    #tester.Test_AI_with_policy(AI_player, game, nn, recurrent_iterations=2)

    print("\n\nLength: " + str(game.length) + "\n")

    return
   

# -----------
# -- STUFF --
# -----------

def play_loop(num_games, game_class, game_args, tester):

    wins = [0,0]
    total_length = 0

    print()
    bar = ChargingBar("Playing", max=num_games, suffix='%(percent)d%% - %(remaining)d')
    bar.next(0)
    for g in range(num_games):
        game = game_class(*game_args)
        winner = tester.random_vs_random(game)
        if winner != 0:
            wins[winner-1] +=1
        total_length += game.length
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

    average_length = total_length / num_games

    print("\n\nLength: " + format(average_length, '.4'))   
    print("P1 Win ratio: " + format(p1_winrate, '.4'))
    print("P2 Win ratio: " + format(p2_winrate, '.4'))
    print("Draw percentage: " + format(draw_percentage, '.4'))
    print("Comparative Win ratio(p1/p2): " + cmp_1_string)
    print("Comparative Win ratio(p2/p1): " + cmp_2_string + "\n", flush=True)

def sanity_test():
    #SANITY TEST

    model = MLP_Network(xor_game())

    network = Torch_NN(model)

    optimizer = torch.optim.Adam(network.get_model().parameters(), lr=0.005)
    
    states = [[[0.0,0.0]],[[0.0,1.0]],[[1.0,0.0]],[[1.0,1.0]]]
    policy = [[[0.5,0.5]], [[1.0,0.0]], [[0.0,1.0]], [[0.5,0.5]]]
    value = [0.0, 1.0, 1.0, 0.0]

    trainining_data = []
    for i in range(len(states)):
        target = (policy[i], value[i])
        pair = (states[i], target)
        trainining_data.append(pair)

    network.get_model().train()

    policy_loss = 0.0
    value_loss = 0.0
    combined_loss = 0

    avarege_loss = 0.0


    epochs = 2000
    for e in range(epochs):
        random.shuffle(trainining_data)

        for (state, (target_policy, target_value)) in trainining_data:
            optimizer.zero_grad()
            state = torch.tensor(state)

            predicted_policy, predicted_value = network.get_model()(state.to(network.device))

            target_policy = torch.tensor(target_policy).to(network.device)
            target_value = torch.tensor(target_value).to(network.device)

        
            #policy_loss += ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) / math.log(len(target_policy)) )
            policy_loss = ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) )
            #Policy loss is being "normalized" by log(num_actions), since cross entropy's expected value is log(target_size)

            #value_loss += ((target_value - predicted_value) ** 2)
            value_loss = torch.abs(target_value - predicted_value)

            combined_loss = policy_loss + value_loss

            combined_loss.backward()
            optimizer.step()

            avarege_loss += combined_loss

        print(avarege_loss/((e+1)*4))

    print("\n\n")
    for (state, (target_policy, target_value)) in trainining_data:
        state = torch.tensor(state)
        predicted_policy, predicted_value = network.inference(state)
        print(state, predicted_policy, predicted_value)

class xor_game():
    def __init__(self):
        self.action_space_shape = (1,1,2)
        self.game_state_shape = (1,1,2)
    
    def get_action_space_shape(self):
        return self.action_space_shape
    
    def state_shape(self):
        return self.game_state_shape

def abv(flag):
    time.sleep(1/20)
    for i in range(200):
        i = i + 2

    return 3

if __name__ == "__main__":
    main()