import os

from AlphaZero import AlphaZero

from Neural_Networks.MLP_Network import MLP_Network as MLP_Net
from Neural_Networks.ConvNet import ConvNet
from Neural_Networks.ResNet import ResNet
from Neural_Networks.RecurrentNet import RecurrentNet

from Agents import Agent
from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.RandomAgent import RandomAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent

from TestManager import TestManager

from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Utils.Functions.general_utils import *
from Utils.Functions.loading_utlis import *
from Utils.Functions.ray_utils import *

from ruamel.yaml import YAML


class Interactive:
    def __init__(self):
        self.yaml_parser = YAML()
        self.yaml_parser.default_flow_style = False
        return


    def start(self):
        print("\nStarted interactive mode!\n")

        print("\nNOTE: This mode is intended to give the user an overview of some of the system\'s functionalities.\
            \nHowever, for more specific use cases, the definition of custom training/testing presets is recommended.\n")
            
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
                raise Exception("Interactive: Option unavailable.")

    #####################################################################################################################
    # ----------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------- TRAINING ------------------------------------------------------ #
    # ----------------------------------------------------------------------------------------------------------------- #
    #####################################################################################################################


    def training_mode(self):
        game_class, game_args = self.choose_game()

        continue_answer = input("\nDo you wish to continue training a previous network or train a new one?(1 or 2)\
                                    \n1 -> Continue training\
                                    \n2 -> Start new training\
                                    \n\nNumber: ")
        
        if continue_answer == "1":
            self.continue_training(game_class, game_args)
        elif continue_answer == "2":
            self.new_training(game_class, game_args)
        else:
            print("Unknown answer.")
            exit()
        return

    def continue_training(self, game_class, game_args):
        game = game_class(*game_args)
        game_name = game.get_name()

        network_name = input("\n\nName of the trained network: ")
        new_network_name = network_name

        new_name_answer = input("\nDo you wish to continue training with a different network name? (y/n): ")
        if self.check_answer(new_name_answer):
            new_network_name = input("\nNew network name: ")
            
        if new_network_name == network_name:
            print("Warning: The previous training configuration will be overwritten to continue training from a checkpoint.")

        network_folder_path = "Games/" + game_name + "/models/" + network_name + "/"
        training_config_path = network_folder_path + "model/train_config_copy.yaml"
        search_config_path = network_folder_path + "model/search_config_copy.yaml"

        train_config = load_yaml_config(self.yaml_parser, training_config_path)
        search_config = load_yaml_config(self.yaml_parser, search_config_path)
        

        train_config["Initialization"]["network_name"] = new_network_name
        train_config["Initialization"]["load_checkpoint"] = True
        train_config["Initialization"]["Checkpoint"]["cp_network_name"] = network_name

        replay_path = network_folder_path + "replay_buffer.cp"
        if os.path.exists(replay_path):
            # If a replay buffer checkpoint exists, we will try to load it
            train_config["Initialization"]["Checkpoint"]["load_buffer"] = True


        generated_training_config_path = "Configs/Config_Files/Training/Interactive/generated_training_config.yaml"
        generated_search_config_path = "Configs/Config_Files/Search/Interactive/generated_search_config.yaml"
        save_yaml_config(self.yaml_parser, generated_training_config_path, train_config)
        save_yaml_config(self.yaml_parser, generated_search_config_path , search_config)


        print("\n")
        context = start_ray_local()
        game_args_list = [game_args]
        alpha_zero = AlphaZero(game_class, game_args_list, generated_training_config_path, generated_search_config_path)
        alpha_zero.run()
        return 
    
    def new_training(self, game_class, game_args):
        game = game_class(*game_args)

        default_train_path = "Configs/Config_Files/Training/Interactive/base_training_config.yaml"
        defaul_search_path = "Configs/Config_Files/Search/Interactive/base_search_config.yaml"

        # Load a default configs and edit them acording to user choices
        train_config = load_yaml_config(self.yaml_parser, default_train_path)
        search_config = load_yaml_config(self.yaml_parser, defaul_search_path)

        network_name = input("\n\nChoose a name for the network: ")
        train_config["Initialization"]["network_name"] = network_name

        model, recurrent = self.choose_new_model(game)
        if recurrent:
            self.train_recurrent_choices(train_config)
        
        softmax_moves = self.search_choices(search_config)
        train_config["Running"]["early_softmax_moves"] = softmax_moves

        self.train_running_choices(train_config)
        self.train_cache_choices(train_config)
        self.train_saving_choices(train_config)
        self.train_testing_choices(train_config)
        self.train_plotting_choices(train_config)
        self.train_learning_choices(train_config)
        self.train_optimizer_scheduler_choices(train_config)


        generated_training_config_path = "Configs/Config_Files/Training/Interactive/generated_training_config.yaml"
        generated_search_config_path = "Configs/Config_Files/Search/Interactive/generated_search_config.yaml"
        save_yaml_config(self.yaml_parser, generated_training_config_path, train_config)
        save_yaml_config(self.yaml_parser, generated_search_config_path , search_config)


        print("\n")
        context = start_ray_local()
        game_args_list = [game_args]
        alpha_zero = AlphaZero(game_class, game_args_list, generated_training_config_path, generated_search_config_path, model=model)
        alpha_zero.run()


    #####################################################################################################################
    # ----------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------- TESTING ---------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------------- #
    #####################################################################################################################

    def testing_mode(self):
        # FIXME: Currently testing in interactive is done by creating the necessary classes directly instead of using testing_configs

        game_class, game_args = self.choose_game()
        game = game_class(*game_args)
        game_name = game.get_name()
        
        p1_agent, p2_agent = self.agent_choices(game_name)

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
            
            print = self.check_answer(input("\nDo you wish to print a game representation to console?(y/n)"))
            slow = self.check_answer(input("\nDo you wish to slow down the game being played?(y/n)"))

            test_manager = TestManager(game_class, game_args, num_actors=1, slow=slow, print=print, render_choice=rendering_mode)
            test_manager.run_visual_test(p1_agent, p2_agent)
        

        elif test_mode_answer == "2":

            num_games = int(input("\nHow many games you wish to play?"))
            num_testers = int(input("\nHow many processes/actors you wish to use?"))


            test_manager = TestManager(game_class, game_args, num_actors=num_testers)
            test_manager.run_test_batch(num_games, p1_agent, p2_agent, False, False, True)


    #####################################################################################################################
    # ----------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------- IMAGES ----------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------------- #
    #####################################################################################################################

    def images_mode(self):
        print("\n\nCurrently not implemented in Interactive.\n")
        return
    

    #####################################################################################################################
    # ----------------------------------------------------------------------------------------------------------------- #
    # ---------------------------------------------------- UTILITY ---------------------------------------------------- #
    # ----------------------------------------------------------------------------------------------------------------- #
    #####################################################################################################################

    def train_recurrent_choices(self, train_config) -> None:
        print("\n\n\n### RECURRENT OPTIONS ###\n\n")
        train_iter = int(input("Number of recurrent iterations to use in inference, during training: "))
        pred_iter = int(input("Number of recurrent iterations to use in inference, during self-play: "))
        test_iter = int(input("NUmber of recurrent iterations to use in inference, during testing: "))
        print("\n\n\n#########################\n\n")

        train_config["Recurrent Options"]["train_iterations"] = [train_iter]
        train_config["Recurrent Options"]["pred_iterations"] = [pred_iter]
        train_config["Recurrent Options"]["test_iterations"] = test_iter

        return
    
    def train_running_choices(self,train_config) -> None:
        print("\n\n\n### RUNTIME OPTIONS ###\n\n")
        print("\nCurrently interative only supports 'sequential' execution mode.\n")

        num_actors = int(input("\n\nHow many actors/processes to use during self-play: "))
        games_per_step = int(input("\n\nHow many games do you wish to play per training step?: "))
        training_steps = int(input("\n\nHow many training steps you wish to run?: "))
        early_fill = int(input("\n\nHow many games you wish to play\nto fill the replay buffer before training starts?\nEarly fill games: "))
        print("\n#######################\n")

        train_config["Running"]["num_actors"] = num_actors
        train_config["Running"]["early_fill_per_type"] = early_fill
        train_config["Running"]["training_steps"] = training_steps
        train_config["Running"]["Sequential"]["num_games_per_type_per_step"] = games_per_step

    def train_cache_choices(self, train_config) -> None:
        print("\n\n\n### CACHE OPTIONS ###\n\n")
        cache_choice = "disabled"
        cache_answer = input("\n\nDo you want to use an inference cache, for AlphaZero's MCTS?(y/n):")
        if self.check_answer(cache_answer):
            cache_choice = self.select_cache()
            max_size = int(input("\n\nMaximum size of the cache (number of entries): "))
        print("\n#####################\n")

        train_config["Cache"]["cache_choice"] = cache_choice
        train_config["Running"]["max_size"] = max_size

    def train_saving_choices(self, train_config) -> None:
        print("\n\n\n### SAVE_CHECKPOINT OPTIONS ###\n\n")
        save_frequency = int(input("\n\nNumber of training steps between each network checkpoint: "))
        save_buffer = self.check_answer(input("\nDo you wish to save replay buffer checkpoints?(y/n) "))
        print("\n###############################\n")
        
        train_config["Saving"]["save_frequency"] = save_frequency
        train_config["Saving"]["save_buffer"] = save_buffer

    def train_testing_choices(self, train_config) -> None:
        print("\n\n\n### TESTING OPTIONS ###\n\n")
        async_testing = self.check_answer(input("\nDo you wish to run the network tests asynchronously(y/n)?"))
        testing_actors = int(input("\nNumber of actors/processes to use during testing: "))
        early_testing = self.check_answer(input("\nDo you wish to test the network before training starts(y/n)?"))
        policy_frequency = int(input("\nNumber of training steps between tests using the policy: "))
        mcts_frequency = int(input("\nNumber of training steps between tests using mcts: "))
        print("\n#######################\n")

        train_config["Testing"]["asynchronous_testing"] = async_testing
        train_config["Testing"]["testing_actors"] = testing_actors
        train_config["Testing"]["early_testing"] = early_testing
        train_config["Testing"]["policy_test_frequency"] = policy_frequency
        train_config["Testing"]["mcts_test_frequency"] = mcts_frequency
        return

    def train_plotting_choices(self, train_config) -> None:
        print("\n\n\n### PLOTTING OPTIONS ###\n\n")
        plot_loss = self.check_answer(input("\nDo you wish to plot loss during training(y/n)?"))
        plot_weights = self.check_answer(input("\nDo you wish to plot information about the network's weights(y/n)?"))
        plot_frequency = int(input("\nNumber of training steps between graph plotting: "))
        print("\n########################\n")

        train_config["Plotting"]["plot_frequency"] = plot_frequency
        train_config["Plotting"]["plot_weights"] = plot_weights
        train_config["Testing"]["plot_loss"] = plot_loss
        return
    
    def train_learning_choices(self, train_config) -> None:
        print("\n\n\n### LEARNING OPTIONS ###\n\n")
        replay_window = int(input("\nMaximum number of games in the replay buffer: "))
        train_config["Learning"]["replay_window_size"] = replay_window

        method_answer = int(input("\nYou want to training the network by epochs or using samples?\n\
                                 1 -> Epochs\n\
                                 2 -> Samples\n\
                                \nNumber: "))
        
        if method_answer == 1:
            train_config["Learning"]["learning_method"] = "epochs"

            learning_epochs = int(input("\n\nNumber of epochs per training_step: "))
            batch_size = int(input("\nBatch size: "))
            plot_epochs = self.check_answer(input("\nDo you want to a graph for the epochs within each training step(y/n)?"))

            train_config["Learning"]["Epochs"]["learning_epochs"] = learning_epochs
            train_config["Learning"]["Epochs"]["batch_size"] = batch_size
            train_config["Learning"]["Epochs"]["plot_epochs"] = plot_epochs
        
        elif method_answer == 2:
            train_config["Learning"]["learning_method"] = "samples"

            batch_size = int(input("\nSample size: "))
            num_samples = int(input("\nNumber of samples: "))
            with_replacement = self.check_answer(input("\nExtract samples with replacement(y/n)?"))
            late_heavy = self.check_answer(input("\nUse destribution that favours the latest games(y/n)?"))

            train_config["Learning"]["Samples"]["batch_size"] = batch_size
            train_config["Learning"]["Samples"]["num_samples"] = num_samples
            train_config["Learning"]["Samples"]["with_replacement"] = with_replacement
            train_config["Learning"]["Samples"]["late_heavy"] = late_heavy
        
        else:
            raise Exception("Learning method not recognized.")


        print("\n########################\n")
        return
    
    def train_optimizer_scheduler_choices(self, train_config) -> None:
        print("\n\n\n### OPTIMIZER AND SCHEDULER OPTIONS ###\n\n")
        optimizer_answer = int(input("\nWhat optimizer do you want to use?\n\
                                 1 -> SGD\n\
                                 2 -> Adam\n\
                                \nNumber: "))

        
        learing_rate = float(input("\nLearning rate: "))
        train_config["Scheduler"]["starting_lr"] = learing_rate

        if optimizer_answer == 1:
            train_config["Optimizer"]["optimizer_choice"] = "SGD"
            momentum_answer = self.check_answer(input("\nDo you wish to use momentum or weight decay(y/n)?"))
            if momentum_answer:
                momentum = float(input("\nMomentum: "))
                weight_decay = float(input("\nWeight decay: "))

            train_config["Optimizer"]["SGD"]["momentum"] = momentum
            train_config["Optimizer"]["SGD"]["weight_decay"] = weight_decay
            
            
        elif optimizer_answer == 2:
            train_config["Optimizer"]["optimizer_choice"] = "Adam"

        else:
            raise Exception("Interactive: Unknown optimizer choice.")

        print("\n#######################################\n")
        return
    
    def select_cache(self) -> str:
        available_caches = ("dict", "keyless")
        print("\nChoose a cache type.\nAvailable caches:")
        for i, cache_name in enumerate(available_caches):
            print(f"\n{i+1} -> {cache_name.capitalize()}")

        cache_index = int(input("\n\nNumber: ")) - 1
        cache_choice = available_caches[cache_index]
        return cache_choice

    def select_model(self) -> str:
        available_models = ("MLP", "ConvNet", "ResNet", "Recurrent")

        model_question = "\nWhat model to you wish to train?(type the number)"
        for i, g in enumerate(available_models):
            model_question += f"\n{i} -> {g}"

        model_question += "\n\nNumber: "
        model_to_use = input(model_question)
        return model_to_use

    def select_game(self) -> str:
        available_games = ("SCS", "tic_tac_toe")

        game_question = "\nWhat game to you wish to play?(type the number):"
        for i,g in enumerate(available_games):
            game_question += f"\n {i} -> {g}"

        game_question += "\n\nNumber: "
        game_to_play = input(game_question)
        return game_to_play

    def search_choices(self, search_config) -> None:
        print("\n\n\n### SEARCH OPTIONS ###\n\n")
        mcts_simulations = int(input("\nNumber of mcts simulations per game move: "))
        softmax_moves = int(input("\nNumber of moves selected using softmax instead of max(type 0 if you always want to choose the greedy move): "))

        search_config["Simulation"]["mcts_simulations"] = mcts_simulations
        search_config["Exploration"]["number_of_softmax_moves"] = softmax_moves
        print("\n#####################\n")
        return softmax_moves

    def agent_choices(self, game_name):
        generic_agents = ("Mcts", "Policy", "Random")
        SCS_agents = ("GoalRush")
        Tic_Tac_Toe_agents = ()
        if game_name == "SCS":
            available_agents = generic_agents + SCS_agents
        elif game_name == "Tic_Tac_Toe":
            available_agents = generic_agents + Tic_Tac_Toe_agents
        
        agent_display = "\nThere are " + str(len(available_agents)) + " types of agents available for this game: "
        for i,a in enumerate(available_agents):
            agent_display += f"\n {i} -> {a} "
        print(agent_display)

        print("\nWe will run tests by pitting two of this agents against each other.")
        print("\"Policy\" and \"Mcts\" agents require a trained network.")
        p1_number = input("\nEnter the number of player one's agent: ")
        p1_agent_name = available_agents[p1_number -1]
        p1_agent = self.create_agent(p1_agent_name, game_name)

        p2_number = input("\nEnter the number of player two's agent: ")
        p2_agent_name = available_agents[p2_number -1]
        p2_agent = self.create_agent(p2_agent_name, game_name)

        return p1_agent, p2_agent

    def create_agent(self, agent_name, game_name) -> Agent:
        agent = None
        print(f"\n{agent_name} agent was chosen.")
        if agent_name in ("Mcts","Policy"):
            print("This agent requires a network.\n")
            network_name = input("\nNetwork name: ")
            checkpoint_number = input("\nWhat is the number of network checkpoint to use?\
                                       (type \'auto\' to use the latest available)\n\
                                      Checkpoint Number:  ")
            if checkpoint_number != "auto":
                checkpoint_number = int(checkpoint_number)
            nn, _, _, _, _, _, _, checkpoint_number = load_network_checkpoint(game_name, network_name, checkpoint_number)
            print(f"\nLoaded \"{network_name}\" network, on checkpoint number {checkpoint_number}\n\n")
            if nn.is_recurrent():
                print("\nThe Loaded network is recurrent.")
                recurrent_iterations = int(input("\nNumber of recurrent iterations to use for the test: "))

            if agent_name == "Mcts":
                print("This agent requires a search configuration.\n")
                search_config_path = input("\nPlease enter the path to the search configuration: ")
                search_config = load_yaml_config(self.yaml_parser, search_config_path)
                use_cache = self.check_answer(input("\nDo you want to use an inference cache for this test(y/n)?"))
                if use_cache:
                    # If the user want to use a cache, we create a simple default cache with around 1000 entries.
                    cache = create_cache("keyless", 1200)
                agent = MctsAgent(search_config, nn, recurrent_iterations, cache)
                
            elif agent_name == "Policy":
                agent = PolicyAgent(nn, recurrent_iterations, cache=None) # cache is not really needed for policy tests
            

        elif agent_name == "GoalRush":
            agent = GoalRushAgent()
        
        elif agent_name == "Random":
            agent = RandomAgent()

        else:
            raise Exception("\nAgent type not recognized.\n")
    
        return agent

    def choose_game(self):
        game_to_play = self.select_game()

        match game_to_play:
            case "SCS":
                game_class = SCS_Game
                game_args = ["Games/SCS/Game_configs/interative_config.yml"]
            case "tic_tac_toe":
                game_class = tic_tac_toe
                game_args = []
            case _:
                raise Exception("\nGame unsupported in interative mode.")

        return game_class, game_args

    def choose_new_model(self, game):
        recurrent = False
        model_to_use = self.select_model()

        hex_answer = input("\n\nWill the model use hexagonal convolutions?(y/n)")
        hexagonal = self.check_answer(hex_answer)

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
                recurrent = True
                in_channels = game.get_state_shape()[0]
                policy_channels = game.get_action_space_shape()[0]

                filters = input("\nNumber of filters to use internally:")      

                model = RecurrentNet(in_channels, policy_channels, int(filters), hex=hexagonal)
                initialize_parameters(model)
                
                    
            case _:
                raise Exception("Model type unsupported in interative mode.")

        return model, recurrent

    def check_answer(self, answer):
        if answer in ("y","Y","yes","YES"):
            return True
        elif answer in ("n","N","no","NO"):
            return False
        else:
            raise Exception("\nInvalid answer: '" + answer + "'.\n")
