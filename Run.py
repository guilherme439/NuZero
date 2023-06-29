import sys
import os, psutil
import gc
import time
import random
import pickle
import ray

from multiprocessing import Pool, Lock, get_context
import multiprocessing

from progress.bar import ChargingBar

from scipy.special import softmax

import numpy as np

import torch
from torch import nn

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network
from Neural_Networks.Simple_Conv_Network import Simple_Conv_Network
from Neural_Networks.ResNet import ResNet

from Neural_Networks.dt_neural_network import *

from SCS.SCS_Renderer import SCS_Renderer

from SCS.SCS_Game import SCS_Game
from SCS.Terrain import Terrain

from Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Alpha_config import Alpha_Zero_config

from AlphaZero import AlphaZero
from Tester import Tester
from RemoteTester import RemoteTester

from SCS.SCS_Renderer import SCS_Renderer



def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    multiprocessing.set_start_method("spawn")

    if (len(sys.argv)!=2):
        print("Run with single mode argument.")
        return
    else:
        mode = int(sys.argv[1])

    match mode:

        case 0:
            print("0")

        case 1:
            game_class = SCS_Game().__class__
            game_args = [[3,0],[0,1]]
            model_class = Simple_Conv_Network(game_class(*game_args)).__class__

            p = input("Choose the AI player: ")
            
            debug_tester = Tester(100, slow=True, debug=True)

            path = "SCS/models/Un_even/Un_even_30_model"
            debug_tester.Test_AI_with_policy(int(p), game_class=game_class, game_args=game_args, model_class=model_class, model_path=path)

        case 2:
            print("\ncaca")
            return

        case 4:
            game = tic_tac_toe()
            game.play_user_vs_user()
            
        case 7:
            game = tic_tac_toe()
            p = input("Choose the AI player: ")

            model_class = MLP_Network(game).__class__

            tester = Tester(5, debug=True)

            tester.ttt_vs_AI_with_policy(int(p), "Tic_Tac_Toe/models/no_square/no_square_200_model", model_class)

        case 8:
            game_class = tic_tac_toe().__class__
            game_args = []
            model_class = TTT_Simple_Network(game_class(*game_args)).__class__

            tester = Tester(2000, show_bar=True, show_results=True)

            path = "Tic_Tac_Toe/models/perfection/perfection_100_model"
            tester.Test_AI_with_policy(1, game_class=game_class, game_args=game_args, model_class=model_class, model_path=path)

        case 9:
            game_class = SCS_Game().__class__
            game_args = [[3,1],[1,2]]

            model = Simple_Conv_Network(game_class(*game_args), num_filters=128)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_SCS_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
            Alpha_Zero.run()
            
        case 10:
            game_class = SCS_Game().__class__
            game_args = [[3,1],[1,2]]

            model = ResNet(game_class(*game_args), num_blocks=2, num_filters=128)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_SCS_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
            Alpha_Zero.run()

        case 11:
            game_class = SCS_Game().__class__
            game_args = [[3,1],[1,2], True]
            game = game_class(*game_args)

            model = dt_net_2d(game, 128)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_SCS_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, True, game_class, game_args, alpha_config, network_name=net_name)
            Alpha_Zero.run()
        
        case 12:
            game_class = tic_tac_toe().__class__
            game_args = []
            
            model = MLP_Network(game_class(*game_args))

            alpha_config = Alpha_Zero_config()
            alpha_config.set_tic_tac_toe_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)

            Alpha_Zero.run()
        
        case 13:
            game_class = tic_tac_toe().__class__
            game_args = []
            
            model = Simple_Conv_Network(game_class(*game_args), kernel_size=(2,2), num_filters=64)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_tic_tac_toe_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
        
            Alpha_Zero.run()

        case 14:
            game_class = tic_tac_toe().__class__
            game_args = []
            
            num_blocks = 2
            kernel_size = (3,3)
            num_filters = 64

            model = ResNet(game_class(*game_args), num_blocks=num_blocks, kernel_size=kernel_size, num_filters=num_filters)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_tic_tac_toe_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
        
            Alpha_Zero.run()

        case 15:

            game_class = tic_tac_toe().__class__
            game_args = []
            game = game_class(*game_args)

            model = dt_net_2d(game, 64)

            alpha_config = Alpha_Zero_config()
            alpha_config.set_tic_tac_toe_config()

            net_name = input("\nSave the network as: ")

            Alpha_Zero = AlphaZero(model, True, game_class, game_args, alpha_config, network_name=net_name)
        
            Alpha_Zero.run()
            return

        case 16:
            game_class = tic_tac_toe().__class__
            game_args = []
            
            pickle_path = "Tic_Tac_Toe/models/fast_conv/Network.pkl"
            model = pickle.load(pickle_path)

            trained_model_path = "Tic_Tac_Toe/models/fast_conv/fast_conv_300_model"
            model.load_state_dict(torch.load(trained_model_path))

            alpha_config = Alpha_Zero_config()
            alpha_config.set_tic_tac_toe_config()

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
            game_args = [[3,1],[1,2], True]
            game = game_class(*game_args)

            model = MLP_Network(game)
            nn = Torch_NN(model, recurrent=False)

            tester = Tester(mcts_simulations=64, pb_c_base=2000, pb_c_init=1.00, use_terminal=True, slow=False, print=False, render=False)
            tester.set_slow_duration(1.3)
            
            #tester.Test_AI_with_mcts(1, game, nn)
            tester.random_vs_random(game)

            print("\n\nLength: " + str(game.length))

            renderer = SCS_Renderer.remote()
            end = renderer.analyse.remote(game)

            ray.get(end) # wait for the rendering to end

        case 20:
            game_class = SCS_Game().__class__
            game_args = [[3,1],[1,2], True]

            tester = Tester(mcts_simulations=64, pb_c_base=2000, pb_c_init=1.00, use_terminal=True, slow=False, print=False, render=False)

            wins = [0,0]
            total_length = 0
            num_games = 5000

            print()
            bar = ChargingBar("Playing", max=num_games)
            for g in range(num_games):
                game = game_class(*game_args)
                winner = tester.random_vs_random(game)
                if winner != 0:
                    wins[winner-1] +=1
                total_length += game.length
                bar.next()
			
            bar.finish

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
            
        case 21:
            pass

        case _:
            print("default")
        
        

    return

# -----------
# -- STUFF --
# -----------

def play_loop():

    game_class = SCS_Game().__class__
    game_args = [[3,1],[1,2], True]

    tester = Tester(mcts_simulations=64, pb_c_base=2000, pb_c_init=1.00, use_terminal=True, slow=False, print=False, render=False)

    wins = [0,0]
    total_length = 0
    num_games = 5000

    print()
    bar = ChargingBar("Playing", max=num_games)
    for g in range(num_games):
        game = game_class(*game_args)
        winner = tester.random_vs_random(game)
        if winner != 0:
            wins[winner-1] +=1
        total_length += game.length
        bar.next()
    
    bar.finish

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