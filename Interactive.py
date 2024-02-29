import os
import time

from Neural_Networks.Torch_NN import Torch_NN
from Neural_Networks.MLP_Network import MLP_Network as MLP_Net

from Neural_Networks.ConvNet import ConvNet
from Neural_Networks.ResNet import ResNet
from Neural_Networks.RecurrentNet import RecurrentNet

from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.RandomAgent import RandomAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent

from Tester import Tester
from RemoteTester import RemoteTester

from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe


def start_interative():
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
        #nn, search_config = load_trained_network(game, net_name, model_iteration)
        
        if rendering_mode == "passive":
            tester = Tester(render=True)
        elif rendering_mode == "interactive":
            tester = Tester(print=True)

        '''
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
        '''

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
        #ray.init()

        game = game_class(*game_args)
        #nn, search_config = load_trained_network(game, net_name, model_iteration)
        
        print("\n\nNeeds to be updated. Currently not working...\n")

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

        #continue_training(game_class, game_args, trained_network_name, continue_network_name, \
        #                       use_same_configs, new_alpha_config_path, new_search_config_path)
        
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

        #alpha_zero = AlphaZero(game_class, game_args, model, network_name, alpha_config_path, search_config_path)
        #alpha_zero.run()

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
            
            model = ConvNet(in_channels, policy_channels, int(kernel_size), int(num_filters), hex=hexagonal)
            

        case "ResNet":
            in_channels = game.get_state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            num_blocks = input("\nNumber of residual blocks: ")
            num_filters = input("Number of filters: ")  
            kernel_size = input("Kernel size (int): ")  

            
            model = ResNet(in_channels, policy_channels, int(num_blocks), int(kernel_size), int(num_filters), hex=hexagonal)
            

        case "Recurrent":
            in_channels = game.get_state_shape()[0]
            policy_channels = game.get_action_space_shape()[0]

            filters = input("\nNumber of filters to use internally:")      

            model = RecurrentNet(in_channels, policy_channels, int(filters), hex=hexagonal)
            
                
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


def images_mode():
    return