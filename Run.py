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



def user_game():
    game = SCS_Game([2,1])
    game.play_user_vs_user()
    return


def main():
    pid = os.getpid()
    process = psutil.Process(pid)

    multiprocessing.set_start_method("spawn")

    if (len(sys.argv)!=2):
        print("Run with single mode argument.")
        return
    else:
        mode = int(sys.argv[1])

    
    if mode == 0:
        user_game()

    elif mode == 1:
        game_class = SCS_Game().__class__
        game_args = [[3,0],[0,1]]
        model_class = Simple_Conv_Network(game_class(*game_args)).__class__

        p = input("Choose the AI player: ")
        
        debug_tester = Tester(100, slow=True, debug=True)

        path = "SCS/models/Un_even/Un_even_30_model"
        debug_tester.Test_AI_with_policy(int(p), game_class=game_class, game_args=game_args, model_class=model_class, model_path=path)

    elif mode == 2:
        print("\ncaca")
        return

    
    elif mode == 4:
        game = tic_tac_toe()
        game.play_user_vs_user()
        
    elif mode == 7:
        game = tic_tac_toe()
        p = input("Choose the AI player: ")

        model_class = MLP_Network(game).__class__

        tester = Tester(5, debug=True)

        tester.ttt_vs_AI_with_policy(int(p), "Tic_Tac_Toe/models/no_square/no_square_200_model", model_class)

    elif mode == 8:
        game_class = tic_tac_toe().__class__
        game_args = []
        model_class = TTT_Simple_Network(game_class(*game_args)).__class__

        tester = Tester(2000, show_bar=True, show_results=True)

        path = "Tic_Tac_Toe/models/perfection/perfection_100_model"
        tester.Test_AI_with_policy(1, game_class=game_class, game_args=game_args, model_class=model_class, model_path=path)

    elif mode == 9:
        game_class = SCS_Game().__class__
        game_args = [[3,1],[1,2]]

        model = Simple_Conv_Network(game_class(*game_args), num_filters=128)

        alpha_config = Alpha_Zero_config()
        alpha_config.set_SCS_config()

        net_name = input("\nSave the network as: ")

        Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
        Alpha_Zero.run()
        
    elif mode == 10:
        game_class = SCS_Game().__class__
        game_args = [[3,1],[1,2]]

        model = ResNet(game_class(*game_args), num_blocks=2, num_filters=128)

        alpha_config = Alpha_Zero_config()
        alpha_config.set_SCS_config()

        net_name = input("\nSave the network as: ")

        Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
        Alpha_Zero.run()

    elif mode == 11:
        game_class = SCS_Game().__class__
        game_args = [[3,1],[1,2]]
        game = game_class(*game_args)

        model = dt_net_2d(game, 128)

        alpha_config = Alpha_Zero_config()
        alpha_config.set_SCS_config()

        net_name = input("\nSave the network as: ")

        Alpha_Zero = AlphaZero(model, True, game_class, game_args, alpha_config, network_name=net_name)
        Alpha_Zero.run()
    
    elif mode == 12:
        game_class = tic_tac_toe().__class__
        game_args = []
        
        model = MLP_Network(game_class(*game_args))

        alpha_config = Alpha_Zero_config()
        alpha_config.set_tic_tac_toe_config()

        net_name = input("\nSave the network as: ")

        Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)

        Alpha_Zero.run()
    
    elif mode == 13:
        game_class = tic_tac_toe().__class__
        game_args = []
        
        model = Simple_Conv_Network(game_class(*game_args), kernel_size=(2,2), num_filters=64)

        alpha_config = Alpha_Zero_config()
        alpha_config.set_tic_tac_toe_config()

        net_name = input("\nSave the network as: ")

        Alpha_Zero = AlphaZero(model, False, game_class, game_args, alpha_config, network_name=net_name)
    
        Alpha_Zero.run()

    elif mode == 14:
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

    elif mode == 15:

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

    elif mode == 16:
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

    elif mode == 17:
        samples = np.random.gamma(1, 0.2, 10)
        random_v = np.random.random(30)

        test_matrix = [[0,1],[0,0],[3,2],[7,1],[4,6]]
        test_list = [0,2,5,1,7,0,-1,1,3]
        test_list_2 = [1,2,5,1,7,0,4,1,3]
        numpy_tensor = np.asarray(test_matrix)
        np_list = np.asarray(test_list)
        list_form = numpy_tensor.flatten()
        test_dict = {}
        test_dict[tuple(list_form)] = "dab"


        np_log_list = np.log(np.asarray(test_list_2))

        print(np_log_list)
        torch_log_list = torch.FloatTensor(np_log_list.astype(np.float64))

        torch_list = torch.FloatTensor(np_list.astype(np.float64))
        print(torch.sum(torch_list * torch_list))

        lib_list = softmax(test_list)
        exp_list = np.exp(test_list)
        my_list = exp_list/np.sum(exp_list)

    elif mode == 18:
        game_class = SCS_Game().__class__
        game_args = [[3,1],[1,2]]
        game = game_class(*game_args)

        model = dt_net_2d(game, 128)

        alpha_config = Alpha_Zero_config()
        alpha_config.set_SCS_config()

        net_name = "Tests"

        Alpha_Zero = AlphaZero(model, True, game_class, game_args, alpha_config, network_name=net_name)
        Alpha_Zero.run()

    elif mode == 19:
        game_class = SCS_Game().__class__
        game_args = [[3,1],[1,2]]
        #game = game_class(*game_args)
        game = tic_tac_toe()

        model = MLP_Network(game)
        nn = Torch_NN(model, recurrent=False)

        tester = Tester(mcts_simulations=64, pb_c_base=2000, pb_c_init=1.00, use_terminal=True, slow=True, print=True, render=False)
        tester.Test_AI_vs_AI_with_policy(game, nn, nn)


    elif mode == 20:
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
    
    elif mode == 21:
        probs = []
        

        print(probs)   
        
        

    return

# -----------
# -- STUFF --
# -----------

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