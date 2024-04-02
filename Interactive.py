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

from TestManager import TestManager

from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Utils.general_utils import *

import ruamel
from ruamel.yaml import YAML


class Interactive:
    def __init__(self):
        self.yaml_parser = YAML()
        self.yaml_parser.default_flow_style = False
        return


    def start(self):
        print("\nStarted interactive mode!\n")

        print("\nNOTE: This mode is intended to give the user an overview of some of the system\'s functionalities.\
            \nHowever, for more specific uses, the definition of training/testing presets is recommended.\n")
            
        mode_answer = input("\nWhat do you wish to do?(insert the number)\
                                \n 1 -> Train a neural network\
                                \n 2 -> Pit two agent against each other\
                                \n 3 -> Create SCS unit counters\
                                \n\nNumber: ")
        
        match int(mode_answer):
            case 1:
                self.training_mode()
            case 2:
                self.testing_mode()
            case 3:
                self.images_mode()

            case _:
                raise Exception("Option unavailable")


    def testing_mode(self):  
        
        game_class, game_args = choose_game()
        game = game_class(*game_args)
        game_name = game.get_name()
        
        p1_agent, p2_agent = choose_agents(game_name)

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
                raise Exception("\nBad rendering answer.")
            
            print_answer = input("\nDo you wish to print a game representation to console?(y/n)")
            print = True if print_answer == "y" else False
            slow_answer = input("\nDo you wish to slow down the game being played?(y/n)")
            slow = True if slow_answer == "y" else False

            test_manager = TestManager(game_class, game_args, num_actors=1, slow=slow, print=print, render_choice=rendering_mode)
            test_manager.run_visual_test(p1_agent, p2_agent)
        

        elif test_mode_answer == "2":

            num_games = int(input("\nHow many games you wish to play?"))
            num_testers = int(input("\nHow many processes/actors you wish to use?"))


            test_manager = TestManager(game_class, game_args, num_actors=num_testers)
            test_manager.run_test_batch(num_games, p1_agent, p2_agent, False, False, True)

    def training_mode(self):
        game_class, game_args = choose_game()
        game = game_class(*game_args)
        game_folder_name = game.get_name()

        continue_answer = input("\nDo you wish to continue training a previous network or train a new one?(1 or 2)\
                                    \n1 -> Continue training\
                                    \n2 -> Start new training\
                                    \n\nNumber: ")
        
        if continue_answer == "1":
            continuing = True
        elif continue_answer == "2":
            continuing = False
        else:
            print("Unknown answer.")
            exit()
                    
        train_config = load_yaml_config()

    def choose_agents(self, game_name):
        generic_agents = ("Mcts", "Policy", "Random")
        SCS_agents = ("GoalRush")
        Tic_Tac_Toe_agents = ()
        if game_name == "SCS":
            available_agents = generic_agents + SCS_agents
        elif game_name == "Tic_Tac_Toe":
            available_agents = generic_agents + Tic_Tac_Toe_agents
        
        agent_display = "\nThere are " + str(len(available_agents)) + " types of agents available for this game: "
        for a in available_agents:
            agent_display += "\n-> " + a
        print(agent_display)

        print("\nWe will run tests by pitting two of this agents against each other.")
        p1_agent_name = input("\nWrite the name of the player one's agent: ")
        p1_agent = agent_choices(p1_agent_name)
        p2_agent_name = input("\nWrite the name of the player two's agent: ")
        p2_agent = agent_choices(p2_agent_name)

        return p1_agent, p2_agent

    def agent_choices(self, agent_name):
        print("\nAgent " + agent_name + " chosen.")
        if agent_name == "Mcts":
            print("This agent requires a network.\n")
            input("\nDo you wish to use a trained network or new one:\
                \n1 -> Trained Network\
                \n2 -> New Network")

    def choose_game(self):
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
                game_args = ["Games/SCS/Game_configs/randomized_config.yml"]
            case "tic_tac_toe":
                game_class = tic_tac_toe
                game_args = []
            case _:
                raise Exception("\nGame unsupported in interative mode.")

        return game_class, game_args

    def choose_new_model(self, game):
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
                raise Exception("Model type unsupported in interative mode.")

        return Torch_NN(model)

    def choose_trained_network(self, game_name):
        network_name = input("\n\nName of the trained network: ")
        model_iteration_answer = input("\nModel iteration number: ")
        model_iteration = int(model_iteration_answer)
        nn = load_network_checkpoint(game_name, network_name, model_iteration)[0]
        return nn

    def images_mode(self):
        print("\n\nCurrently unavailable.\n")
        return