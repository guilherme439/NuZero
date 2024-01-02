import os
import psutil
import gc
import resource
import random
import pickle
import time
import sys
import ray
import glob
import re
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from copy import deepcopy
from random import randrange
import more_itertools

from Neural_Networks.Torch_NN import Torch_NN

from Configs.Training_Config import Training_Config
from Configs.Search_Config import Search_Config

from Gamer import Gamer
from ReplayBuffer import ReplayBuffer
from RemoteStorage import RemoteStorage
from RemoteTester import RemoteTester
from TestManager import TestManager
from RemoteTestManager import RemoteTestManager

from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.MctsAgent import MctsAgent
from Agents.Generic.RandomAgent import RandomAgent

from Utils.stats_utilities import *
from Utils.loss_functions import *
from Utils.PrintBar import PrintBar

from progress.bar import ChargingBar
from progress.spinner import PieSpinner




class AlphaZero():

	
    def __init__(self, game_class, game_args_list, model, net_name, train_config_path, search_config_path, plot_data_path=None, state_set=None):

        
        # ------------------------------------------------------ #
        # -------------------- SYSTEM SETUP -------------------- #
        # ------------------------------------------------------ #

        self.game_args_list = game_args_list  # Args for the game's __init__()
        self.game_class = game_class

        current_directory = os.getcwd()
        print("\nCurrent working directory: " + str(current_directory))

        self.network_name = net_name

        self.game_folder_name = self.game_class().get_name()
        self.model_folder_path = self.game_folder_name + "/models/" + self.network_name + "/"
        if not os.path.exists(self.model_folder_path):
            os.mkdir(self.model_folder_path)

        self.plots_path = self.model_folder_path + "plots/"
        if not os.path.exists(self.plots_path):
            os.mkdir(self.plots_path)

        
        self.plot_data_save_path = self.model_folder_path + "plot_data.pkl"
        self.plot_data_load_path = plot_data_path

        self.latest_network = Torch_NN(model)
            
        self.search_config = Search_Config()
        self.search_config.load(search_config_path)

        self.train_config = Training_Config()
        self.train_config.load(train_config_path)
        
        self.state_set = state_set

        # ------------------------------------------------------ #
        # ----------------------- PLOTS ------------------------ #
        # ------------------------------------------------------ #

        self.train_global_value_loss = []
        self.train_global_policy_loss = []
        self.train_global_combined_loss = []

        self.epochs_value_loss = []
        self.epochs_policy_loss = []
        self.epochs_combined_loss = []

        self.p1_policy_wr_stats = [[],[]]
        self.p2_policy_wr_stats = [[],[]]
        self.p1_mcts_wr_stats = [[],[]]
        self.p2_mcts_wr_stats = [[],[]]

        self.weight_size_max = []
        self.weight_size_min = []
        self.weight_size_average = []

        if self.state_set is not None:
            self.state_set_stats = [ [] for state in self.state_set ]


        

    def run(self, starting_iteration=0):
        pid = os.getpid()
        process = psutil.Process(pid)

        # ------------------------------------------------------ #
        # ------------------ RUNTIME CONFIGS ------------------- #
        # ------------------------------------------------------ #

        print("\n\n--------------------------------\n")

        running_mode = self.train_config.running["running_mode"]
        num_actors = self.train_config.running["num_actors"]
        early_fill_games_per_type = self.train_config.running["early_fill_per_type"]

        training_steps = int(self.train_config.running["training_steps"])
        self.num_game_types = len(self.game_args_list)
        if running_mode == "asynchronous":
            if self.num_game_types > 1:
                print("Asynchronous mode does not support training with multiple games.\nExiting.")
                exit()
            update_delay = self.train_config.asynchronous["update_delay"]
        elif running_mode == "sequential":
            num_games_per_type_per_step = self.train_config.sequential["num_games_per_type_per_step"]
            

        cache_choice = self.train_config.cache["cache_choice"]
        size_estimate = self.train_config.cache["size_estimate"]

        save_frequency = self.train_config.saving["save_frequency"]
        storage_frequency = self.train_config.saving["storage_frequency"]

        train_iterations = self.train_config.recurrent_training["train_iterations"]
        pred_iterations = self.train_config.recurrent_training["pred_iterations"]
        prog_alpha = self.train_config.recurrent_training["alpha"]

        asynchronous_testing = self.train_config.testing["asynchronous_testing"]
        num_testers = self.train_config.testing["testing_actors"]
        early_testing = self.train_config.testing["early_testing"]
        policy_test_frequency = self.train_config.testing["policy_test_frequency"]
        mcts_test_frequency = self.train_config.testing["mcts_test_frequency"]
        num_policy_test_games = self.train_config.testing["num_policy_test_games"]
        num_mcts_test_games = self.train_config.testing["num_mcts_test_games"]
        test_iterations = self.train_config.testing["test_iterations"]
        test_game_index = self.train_config.testing["test_game_index"]
        
        plot_frequency = self.train_config.plotting["plot_frequency"]
        policy_split = self.train_config.plotting["policy_split"]
        value_split = self.train_config.plotting["value_split"]
        combined_split = self.train_config.plotting["combined_split"]
        self.plot_loss = self.train_config.plotting["plot_loss"]
        self.plot_weights = self.train_config.plotting["plot_weights"]

        if running_mode == "asynchronous":
            asynchronous_testing = True
            # When running asynchronously, tests need to also be async.
        
        
        # ------------------------------------------------------ #
        # ------------------- BACKUP FILES --------------------- #
        # ------------------------------------------------------ #
        
        # dummy forward pass to initialize the weights

        game = self.game_class(*self.game_args_list[0])
        self.latest_network.inference(game.generate_state_image(), False, 1)

        # write model summary and game args to file
        file_name = self.model_folder_path + "model_and_game_config.txt"
        with open(file_name, "w") as file:
            file.write(self.game_args_list.__str__())
            file.write("\n\n\n\n----------------------------------\n\n")
            file.write(self.latest_network.get_model().__str__())
            

        # pickle the network class
        file_name = self.model_folder_path + "base_model.pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(self.latest_network.get_model().cpu(), file)
            print(f'Successfully pickled model class at "{file_name}".\n')

        # create copies of the config files
        print("\nCreating config file copies:")
        search_config_copy_path = self.model_folder_path + "search_config_copy.ini"
        train_config_copy_path = self.model_folder_path + "train_config_copy.ini"
        self.train_config.save(train_config_copy_path)
        self.search_config.save(search_config_copy_path)
        print("\n\n--------------------------------\n")

        # ------------------------------------------------------ #
        # ------------- STORAGE AND BUFFERS SETUP -------------- #
        # ------------------------------------------------------ #

        shared_storage_size = self.train_config.learning["shared_storage_size"]
        replay_window_size = self.train_config.learning["replay_window_size"]
        learning_method = self.train_config.learning["learning_method"]

        self.network_storage = RemoteStorage.remote(shared_storage_size)
        self.latest_network.model_to_cpu()
        ray.get(self.network_storage.store.remote(self.latest_network))
        self.latest_network.model_to_device()

        plot_epochs = False
        if learning_method == "epochs":
            batch_size = self.train_config.epochs["batch_size"]
            learning_epochs = self.train_config.epochs["learning_epochs"]
            plot_epochs = self.train_config.epochs["plot_epoch"]	
            if plot_epochs:
                self.epochs_path = self.plots_path + "Epochs/"
                if not os.path.exists(self.epochs_path):
                    os.mkdir(self.epochs_path)
        elif learning_method == "samples":
            batch_size = self.train_config.samples["batch_size"]

        self.replay_buffer = ReplayBuffer.remote(replay_window_size, batch_size)
            

        # ------------------------------------------------------ #
        # ------------------ OPTIMIZER SETUP ------------------- #
        # ------------------------------------------------------ #

        optimizer_name = self.train_config.optimizer["optimizer"]
        learning_rate = self.train_config.optimizer["learning_rate"]

        weight_decay = self.train_config.optimizer["weight_decay"]
        momentum = self.train_config.optimizer["momentum"]
        nesterov = self.train_config.optimizer["nesterov"]

        scheduler_boundaries = self.train_config.optimizer["scheduler_boundaries"]
        scheduler_gamma = self.train_config.optimizer["scheduler_gamma"]
        
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=learning_rate)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.latest_network.get_model().parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        else:
            optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=learning_rate)
            print("Bad optimizer config.\nUsing default optimizer (Adam)...")

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)

        # ------------------------------------------------------ #
        # --------------------- ALPHAZERO ---------------------- #
        # ------------------------------------------------------ #

        test_game_args = self.game_args_list[test_game_index]
        if asynchronous_testing:
            self.test_futures = []
            self.test_manager = RemoteTestManager.remote(self.game_class, test_game_args, num_testers,
                                                        self.network_storage, self.state_set)
        else:
            self.test_manager = TestManager(self.game_class, test_game_args, num_testers,
                                            self.network_storage, self.state_set)

        if running_mode == "sequential":
            self.games_per_step = num_games_per_type_per_step * self.num_game_types
            print("\nRunning for " + str(training_steps) + " training steps with " + str(self.games_per_step) + " games in each step.")
        elif running_mode == "asynchronous":
            print("\nRunning for " + str(training_steps) + " training steps with " + str(update_delay) + "s of delay between each step.")
        if early_fill_games_per_type > 0:
            total_early_fill = early_fill_games_per_type * self.num_game_types
            print("\n-Playing " + str(total_early_fill) + " initial games to fill the replay buffer.")
        if cache_choice != "disabled":
            print("\n-Using cache for inference results.")			  
        if starting_iteration != 0:
            print("\n-Starting from iteration " + str(starting_iteration+1) + ".\n")

        model = self.latest_network.get_model()
        model_dict = model.state_dict()

        print("\n\n--------------------------------\n")

        if starting_iteration != 0:
            print("NOTE: when continuing training both optimizer state and replay buffer are reset.\n")
            self.load_data(self.plot_data_load_path)
        
        else:
            # Initial save (untrained network)
            save_path = self.model_folder_path + self.network_name + "_" + str(starting_iteration) + "_model"
            torch.save(model_dict, save_path)

            if self.plot_weights:
                self.update_weight_data()
                self.plot_weight()
            
            if early_testing: # For graphing purposes
                test_policy = (policy_test_frequency != 0)
                test_mcts = (mcts_test_frequency != 0)
                policy_games = num_policy_test_games if test_policy else 0
                mcts_games = num_mcts_test_games if test_mcts else 0
                self.run_tests(policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, size_estimate)
                print("\nLaunched early tests.")

        if early_fill_games_per_type > 0:
            print("\n\n\n\nEarly Buffer Fill\n")
            self.run_selfplay(early_fill_games_per_type, cache_choice, size_estimate=size_estimate, text="Playing initial games", early_fill=True)

        if running_mode == "asynchronous":
            actor_list= [Gamer.options(max_concurrency=2).remote
                        (
                        self.replay_buffer,
                        self.network_storage,
                        self.game_class,
                        self.game_args_list[0],
                        0,
                        self.search_config,
                        pred_iterations[0],
                        cache_choice,
                        size_estimate
                        )
                        for a in range(num_actors)]
            
            termination_futures = [actor.play_forever.remote() for actor in actor_list]

        # ---- MAIN TRAINING LOOP ---- #
        steps_to_run = range(starting_iteration, training_steps)
        for step in steps_to_run:
            print("\n\n\n\nStep: " + str(step+1) + "\n")

            if running_mode == "sequential":
                self.run_selfplay(num_games_per_type_per_step, cache_choice, size_estimate=size_estimate, text="Self-Play Games")

            print("\n\nLearning rate: " + str(scheduler.get_last_lr()[0]))
            self.train_network(optimizer, scheduler, batch_size, learning_method, prog_alpha)

            # ---- TESTS ---- #
            test_policy = (policy_test_frequency and (((step+1) % policy_test_frequency) == 0)) 
            test_mcts = (mcts_test_frequency and (((step+1) % mcts_test_frequency) == 0))
            policy_games = num_policy_test_games if test_policy else 0
            mcts_games = num_mcts_test_games if test_mcts else 0
            if asynchronous_testing:
                if len(self.test_futures) > 0:
                    futures, types = list(zip(*self.test_futures))
                    futures = list(futures)
                    (futures_ready, remaining_futures) = ray.wait(futures, timeout=0.1)
                    print("Awaiting results of " + str(len(remaining_futures)) + " test(s)\n")
                    for future in futures_ready:
                        i = futures.index(future)
                        test_type = types[i]
                        result = ray.get(future)
                        self.update_wr_data(result, test_type)
                        self.test_futures.pop(i)
                
            if policy_games or mcts_games:
                self.run_tests(policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, size_estimate)

            # ---- PLOTS ---- #
            # The main thread is responsible for doing the graphs since matplotlib crashes when it runs outside the main thread
            if plot_frequency and (((step+1) % plot_frequency) == 0):
                
                if self.plot_weights:
                    self.update_weight_data()
                    self.plot_weight()
                
                if self.state_set is not None:
                    self.update_state_set_data(test_iterations)
                    self.plot_state_set()

                if self.plot_loss:
                    self.plot_global_loss()
                    if plot_epochs and learning_epochs>1:
                        self.plot_epoch_loss()
                    
                self.plot_wr()

            if policy_split and ((step+1) == policy_split):
                self.split_policy_loss_graph()

            if value_split and ((step+1) == value_split):
                self.split_value_loss_graph()

            if combined_split and ((step+1) == combined_split):
                self.split_combined_loss_graph()
                    
            if save_frequency and (((step+1) % save_frequency) == 0):
                save_path = self.model_folder_path + self.network_name + "_" + str(step+1) + "_model"
                torch.save(self.latest_network.get_model().state_dict(), save_path)
            
            if storage_frequency and (((step+1) % storage_frequency) == 0):
                self.latest_network.model_to_cpu()
                ray.get(self.network_storage.store.remote(self.latest_network))
                self.latest_network.model_to_device()

            # Save plotting data in case of a crash
            self.save_data(self.plot_data_save_path)

            if running_mode == "asynchronous":
                divisions = 10
                small_rest = update_delay/divisions
                sleep_bar = PrintBar('Sleeping', divisions, 15)
                for i in range(divisions):
                    time.sleep(small_rest)
                    sleep_bar.next()
                sleep_bar.finish()

            print("-------------------------------------\n")
            print("\nMain process memory usage: ")
            print("Current memory usage: " + format(process.memory_info().rss/(1024*1000), '.6') + " MB") 
            print("Peak memory usage:    " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, '.6') + " MB\n" )
            print("\n-------------------------------------\n\n")
            # psutil gives memory in bytes and resource gives memory in kb (1024 bytes)

        if asynchronous_testing:
            if len(self.test_futures) > 0:
                bar = PrintBar("Finishing tests", len(self.test_futures), 15)
                for future, i in self.test_futures:
                    result = ray.get(future)
                    self.update_wr_data(result, i)
                    bar.next()

                bar.finish()


            self.plot_wr()
            print("All tests done.\n")
        
        # If you don't wish to wait for the actors to terminate their games,
        # you can comment all the code under this line.
        if running_mode == "asynchronous":
            print("Waiting for actors to finish their games\n")
            for actor in actor_list:
                actor.stop.remote() # tell the actors to stop playing

            ray.get(termination_futures) # wait for each of the actors to terminate the game that they are currently playing       
        
        print("All done.\nExiting")
        return
            
    def run_selfplay(self, num_games_per_type, cache_choice, size_estimate=10000, text="Self-Play", early_fill=False):
        start = time.time()
        stats_list = []

        pred_iterations_list = self.train_config.recurrent_training["pred_iterations"]
        num_actors = self.train_config.running["num_actors"]

        search_config = deepcopy(self.search_config)
        if early_fill:
            softmax_moves = self.train_config.running["early_softmax_moves"]
            softmax_exploration = self.train_config.running["early_softmax_exploration"]
            random_exploration = self.train_config.running["early_random_exploration"]
            search_config.exploration["number_of_softmax_moves"] = softmax_moves
            search_config.exploration["epsilon_softmax_exploration"] = softmax_exploration
            search_config.exploration["epsilon_random_exploration"] = random_exploration

        total_games = self.num_game_types * num_games_per_type
        bar = PrintBar(text, total_games, 15)
        for i in range(len(self.game_args_list)):
            game_args = self.game_args_list[i]
            iterations = pred_iterations_list[i]
            game_index = i
            actor_list= [Gamer.remote
                        (
                        self.replay_buffer,
                        self.network_storage,
                        self.game_class,
                        game_args,
                        game_index,
                        search_config,
                        iterations,
                        cache_choice,
                        size_estimate
                        )
                        for a in range(num_actors)]
                
            actor_pool = ray.util.ActorPool(actor_list)

            call_args = []
            for g in range(num_games_per_type):
                actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)
            
            time.sleep(1) # Sometimes ray bugs if we dont wait before getting the results
            
            for g in range(num_games_per_type):
                stats = actor_pool.get_next_unordered()
                stats_list.append(stats)
                bar.next()

        bar.finish()
        print_stats_list(stats_list)

        end = time.time()
        total_time = end-start
        print("\n\nTotal time(m): " + format(total_time/60, '.4'))
        print("Average time per game(s): " + format(total_time/total_games, '.4'))

        return

    def run_tests(self, policy_games, mcts_games, test_iterations, asynchronous_testing, cache_choice, size_estimate=10000):
        latest_network = ray.get(self.network_storage.get.remote()) # ask for a copy of the latest network
        
        if mcts_games:
            mcts_agent = MctsAgent(self.search_config, latest_network, test_iterations, cache_choice, size_estimate)
        if policy_games:
            policy_agent = PolicyAgent(latest_network, test_iterations)

        random_agent = RandomAgent()

        p1_policy = p2_policy = p1_mcts = p2_mcts = None

        if asynchronous_testing:
            if policy_games:
                p1_policy = self.test_manager.run_test_batch.remote(policy_games, policy_agent, random_agent, show_results=True)
                p2_policy = self.test_manager.run_test_batch.remote(policy_games, random_agent, policy_agent, show_results=True)
            if mcts_games:
                p1_mcts = self.test_manager.run_test_batch.remote(mcts_games, mcts_agent, random_agent, show_results=True)
                p2_mcts = self.test_manager.run_test_batch.remote(mcts_games, random_agent, mcts_agent, show_results=True)
            
            futures = [p1_policy, p2_policy, p1_mcts, p2_mcts]
            for i in range(len(futures)):
                future = futures[i]
                if future is not None:
                    self.test_futures.append((future, i))
                    
        else:
            if policy_games:
                p1_policy = self.test_manager.run_test_batch(policy_games, policy_agent, random_agent, show_results=True)
                p2_policy = self.test_manager.run_test_batch(policy_games, random_agent, policy_agent, show_results=True)
            if mcts_games:
                p1_mcts = self.test_manager.run_test_batch(mcts_games, mcts_agent, random_agent, show_results=True)
                p2_mcts = self.test_manager.run_test_batch(mcts_games, random_agent, mcts_agent, show_results=True)

            results = [p1_policy, p2_policy, p1_mcts, p2_mcts]
            for i in range(len(results)):
                result = results[i]
                if result is not None:
                    self.update_wr_data(result, i)

        return
        
    def train_network(self, optimizer, scheduler, batch_size, learning_method, prog_alpha):
        '''Executes a training step'''
        print()
        start = time.time()

        replace = True
        train_iterations = self.train_config.recurrent_training["train_iterations"]
        batch_extraction = self.train_config.learning["batch_extraction"]

        replay_size = ray.get(self.replay_buffer.len.remote(), timeout=120)
        n_games = ray.get(self.replay_buffer.played_games.remote(), timeout=120)

        print("\nAfter Self-Play there are a total of " + str(replay_size) + " positions in the replay buffer.")
        print("Total number of games: " + str(n_games))
        

        # ----------- Loss functions -----------

        value_loss_choice = self.train_config.learning["value_loss"]
        policy_loss_choice = self.train_config.learning["policy_loss"]
        normalize_CEL = self.train_config.learning["normalize_cel"]

        normalize_policy = False
        match policy_loss_choice:
            case "CEL":
                policy_loss_function = nn.CrossEntropyLoss(label_smoothing=0.02)
                if normalize_CEL:
                    normalize_policy = True
            case "KLD":
                policy_loss_function = KLDivergence
            case "MSE":
                policy_loss_function = MSError
        
        match value_loss_choice:
            case "SE":
                value_loss_function = SquaredError
            case "AE":
                value_loss_function = AbsoluteError
        
        # --------------------------------------

        
        if learning_method == "epochs":
            learning_epochs = self.train_config.epochs["learning_epochs"]
            plot_epoch = self.train_config.epochs["plot_epoch"]	
            
            if  batch_size > replay_size:
                print("Batch size too large.\n" + 
                    "If you want to use batch_size with more moves than the first batch of games played " + 
                    "you need to use, the \"early_fill\" config to fill the replay buffer with random games at the start.\n")
                exit()
            else:
                number_of_batches = replay_size // batch_size
                print("Batches in replay buffer: " + str(number_of_batches))

                print("Batch size: " + str(batch_size))	
                print("\n")

            value_loss = 0.0
            policy_loss = 0.0
            combined_loss = 0.0

            if self.plot_loss:
                self.epochs_value_loss.clear()
                self.epochs_policy_loss.clear()
                self.epochs_combined_loss.clear()
            
            total_updates = learning_epochs*number_of_batches
            print("\nTotal number of updates: " + str(total_updates) + "\n")
            
            #bar = ChargingBar('Training ', max=learning_epochs)
            bar = PrintBar('Training step ', learning_epochs, 15)
            for e in range(learning_epochs):

                ray.get(self.replay_buffer.shuffle.remote(), timeout=120) # ray.get() beacuse we want the shuffle to be done before using buffer
                if batch_extraction == 'local':
                    future_replay_buffer = self.replay_buffer.get_buffer.remote()

                epoch_value_loss = 0.0
                epoch_policy_loss = 0.0
                epoch_combined_loss = 0.0

                #spinner = PieSpinner('\t\t\t\t\t\t  Running epoch ')
                if batch_extraction == 'local':
                    # We get entire buffer and slice locally to avoid a lot of remote calls
                    replay_buffer = ray.get(future_replay_buffer, timeout=300) 

                for b in range(number_of_batches):		
                    start_index = b*batch_size
                    next_index = (b+1)*batch_size

                    if batch_extraction == 'local':
                        batch = replay_buffer[start_index:next_index]
                    else:
                        batch = ray.get(self.replay_buffer.get_slice.remote(start_index, next_index))
                    
                
                    value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler,
                                                                policy_loss_function, value_loss_function, normalize_policy,
                                                                batch, train_iterations, prog_alpha)

                    epoch_value_loss += value_loss
                    epoch_policy_loss += policy_loss
                    epoch_combined_loss += combined_loss
                    #spinner.next()

                epoch_value_loss /= number_of_batches
                epoch_policy_loss /= number_of_batches
                epoch_combined_loss /= number_of_batches	

                if self.plot_loss:
                    self.epochs_value_loss.append(epoch_value_loss)
                    self.epochs_policy_loss.append(epoch_policy_loss)
                    self.epochs_combined_loss.append(epoch_combined_loss)


                #spinner.finish()
                bar.next()
                
            bar.finish()

            self.train_global_value_loss.extend(self.epochs_value_loss)
            self.train_global_policy_loss.extend(self.epochs_policy_loss)
            self.train_global_combined_loss.extend(self.epochs_combined_loss)
            

                
        elif learning_method == "samples":
            num_samples = self.train_config.samples["num_samples"]
            late_heavy = self.train_config.samples["late_heavy"]

            if batch_extraction == 'local':
                future_buffer = self.replay_buffer.get_buffer.remote()

            batch = []
            probs = []
            if late_heavy:
                # The way I found to create a scalling array
                variation = 0.5 # number between 0 and 1
                num_positions = replay_size
                offset = (1-variation)/2    
                fraction = variation / num_positions

                total = offset
                for _ in range(num_positions):
                    total += fraction
                    probs.append(total)

                probs /= np.sum(probs)

            average_value_loss = 0
            average_policy_loss = 0
            average_combined_loss = 0

            print("\nTotal number of updates: " + str(num_samples) + "\n")
            if batch_extraction == 'local':
                replay_buffer = ray.get(future_buffer, timeout=300)

            #bar = ChargingBar('Training ', max=num_samples)
            bar = PrintBar('Training step', num_samples, 15)
            for _ in range(num_samples):
                if batch_extraction == 'local':
                    if len(probs) == 0:
                        args = [len(replay_buffer), batch_size, replace]
                    else:
                        args = [len(replay_buffer), batch_size, replace, probs]
                    
                    batch_indexes = np.random.choice(*args)
                    batch = [replay_buffer[i] for i in batch_indexes]
                else:
                    batch = ray.get(self.replay_buffer.get_sample.remote(batch_size, replace, probs))

                value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler,
                                                                policy_loss_function, value_loss_function, normalize_policy,
                                                                batch, train_iterations, prog_alpha)

                average_value_loss += value_loss
                average_policy_loss += policy_loss
                average_combined_loss += combined_loss

                bar.next()

            bar.finish()

            average_value_loss /= num_samples
            average_policy_loss /= num_samples
            average_combined_loss /= num_samples

        
            self.train_global_value_loss.extend([average_value_loss])
            self.train_global_policy_loss.extend([average_policy_loss])
            self.train_global_combined_loss.extend([average_combined_loss])

        else:
            print("Bad learning_method config.\nExiting")
            exit()

        
        end = time.time()
        total_time = end-start
        print("\n\nTraining time(s): " + format(total_time, '.4') + "\n\n\n")

        return	

    def batch_update_weights(self, optimizer, scheduler, policy_loss_function, value_loss_function, normalize_policy, batch, train_iterations, alpha):
        '''updates network's weights based on loss values'''

        self.latest_network.get_model().train()
        optimizer.zero_grad()

        loss = 0.0
        value_loss, policy_loss, combined_loss = 0.0, 0.0, 0.0

        if self.latest_network.get_model().recurrent:
            data_by_game = more_itertools.bucket(batch, key=lambda x: x[2]) 
            for index in sorted(data_by_game):
                batch_data = list(data_by_game[index])
                batch_size = len(batch_data)

                states, targets, indexes = list(zip(*batch_data))
                batch_input = torch.cat(states, 0)

                recurrent_iterations = train_iterations[index]
                
                total_value_loss, total_policy_loss, total_combined_loss = 0.0, 0.0, 0.0
                prog_value_loss, prog_policy_loss, prog_combined_loss = 0.0, 0.0, 0.0
                if alpha != 1:
                    outputs, _ = self.latest_network.inference(batch_input, True, recurrent_iterations)
                    total_value_loss, total_policy_loss, total_combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                                                policy_loss_function, value_loss_function, normalize_policy)

                if alpha != 0:
                    outputs = self.get_output_for_prog_loss(batch_input, recurrent_iterations)
                    prog_value_loss, prog_policy_loss, prog_combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                                                policy_loss_function, value_loss_function, normalize_policy)
                
                value_loss = (1 - alpha) * total_value_loss + alpha * prog_value_loss
                policy_loss = (1 - alpha) * total_policy_loss + alpha * prog_policy_loss
                combined_loss = (1 - alpha) * total_combined_loss + alpha * prog_combined_loss

    
        else:
            outputs = self.latest_network.inference(batch_input, True, train_iterations)
            value_loss, policy_loss, combined_loss = self.calculate_loss(outputs, targets, batch_size,
                                                                        policy_loss_function, value_loss_function, normalize_policy)
        
        
        loss = combined_loss

        loss.backward()
        optimizer.step()
        scheduler.step()
        

        return value_loss.item(), policy_loss.item(), combined_loss.item()
    
    def calculate_loss(self, outputs, targets, batch_size, policy_loss_function, value_loss_function, normalize_policy):
        target_values, target_policies = list(zip(*targets))
    
        predicted_policies, predicted_values = outputs

        policy_loss = 0.0
        value_loss = 0.0
        combined_loss = 0.0

        for i in range(batch_size):
            
            target_policy = torch.tensor(target_policies[i]).to(self.latest_network.device)
            target_value = torch.tensor(target_values[i]).to(self.latest_network.device)

            predicted_value = predicted_values[i]
            predicted_policy = predicted_policies[i]
            predicted_policy = torch.flatten(predicted_policy)

            policy_loss += policy_loss_function(predicted_policy, target_policy)
            value_loss += value_loss_function(predicted_value, target_value)
            
        # Policy loss is "normalized" by log(num_actions), since cross entropy's expected value grows with log(target_size)
        target_size = len(targets)
        if normalize_policy:
            policy_loss /= math.log(target_size)

        value_loss /= batch_size
        policy_loss /= batch_size


        combined_loss = policy_loss + value_loss

        invalid_loss = False
        if torch.any(torch.isnan(value_loss)):
            print("\nValue Loss is nan.")
            invalid_loss = True

        if torch.any(torch.isnan(policy_loss)):
            print("\nPolicy Loss is nan.")
            invalid_loss = True
        
        if invalid_loss:
            print("\n\n")
            print(predicted_values)
            print("\n\n")
            print(predicted_policies)
            exit()

        return value_loss, policy_loss, combined_loss

    def get_output_for_prog_loss(self, inputs, max_iters):
        # get features from n iterations to use as input
        n = randrange(0, max_iters)

        # do k iterations using intermediate features as input
        k = randrange(1, max_iters - n + 1)

        if n > 0:
            _, interim_thought = self.latest_network.inference(inputs, True, iters_to_do=n)
            interim_thought = interim_thought.detach()
        else:
            interim_thought = None

        outputs, _ = self.latest_network.inference(inputs, True, iters_to_do=k, interim_thought=interim_thought)
        return outputs

    ##########################################################################
    # -----------------------------            ----------------------------- #
    # ---------------------------    PLOTTING    --------------------------- #
    # -----------------------------            ----------------------------- #
    ##########################################################################

    def plot_epoch_loss(self, step_number):
        print("\nPlotting epochs...")
        plt.plot(range(self.epoch_value_loss), self.epoch_value_loss)
        plt.title("Epoch value loss")
        plt.legend()
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Value_loss.png')
        plt.clf()

        plt.plot(range(self.epoch_policy_loss), self.epcoh_policy_loss)
        plt.title("Epoch policy loss")
        plt.legend()
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Policy_loss.png')
        plt.clf()

        plt.plot(range(self.epoch_combined_loss), self.epoch_combined_loss)
        plt.title("Epoch combined loss")
        plt.legend()
        plt.savefig(self.epochs_path + "step_" + str(step_number) + '-Combined_loss.png')
        plt.clf()
        print("Epoch plotting done.\n")

    def plot_global_loss(self):
        print("\nPlotting global loss...")
        num_points = len(self.train_global_value_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_value_loss, label = "Training")
            plt.title("Global Value loss")
            plt.legend()
            plt.savefig(self.plots_path + '_global_value_loss.png')
            plt.clf()

        num_points = len(self.train_global_policy_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_policy_loss, label = "Training")
            plt.title("Global Policy loss")
            plt.legend()
            plt.savefig(self.plots_path + '_global_policy_loss.png')
            plt.clf()

        num_points = len(self.train_global_combined_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_combined_loss, label = "Training")
            plt.title("Global Combined loss")
            plt.legend()
            plt.savefig(self.plots_path + '_global_total_loss.png')
            plt.clf()

        print("Global loss plotting done.\n")

    def plot_wr(self):
        print("\nPloting wr graphs...")
        if len(self.p1_policy_wr_stats[0]) > 1:
            plt.plot(range(len(self.p1_policy_wr_stats[1])), self.p1_policy_wr_stats[1], label = "P2")
            plt.plot(range(len(self.p1_policy_wr_stats[0])), self.p1_policy_wr_stats[0], label = "P1")
            plt.title("Policy -> Win rates as Player 1")
            plt.legend()
            plt.savefig(self.plots_path + 'p1_policy_wr.png')
            plt.clf()

        if len(self.p2_policy_wr_stats[0]) > 1:
            plt.plot(range(len(self.p2_policy_wr_stats[0])), self.p2_policy_wr_stats[0], label = "P1")
            plt.plot(range(len(self.p2_policy_wr_stats[1])), self.p2_policy_wr_stats[1], label = "P2")
            plt.title("Policy -> Win rates as Player 2")
            plt.legend()
            plt.savefig(self.plots_path + 'p2_policy_wr.png')
            plt.clf()

        if len(self.p1_mcts_wr_stats[0]) > 1:
            plt.plot(range(len(self.p1_mcts_wr_stats[1])), self.p1_mcts_wr_stats[1], label = "P2")
            plt.plot(range(len(self.p1_mcts_wr_stats[0])), self.p1_mcts_wr_stats[0], label = "P1")
            plt.title("MCTS -> Win rates as Player 1")
            plt.legend()
            plt.savefig(self.plots_path + 'p1_mcts_wr.png')
            plt.clf()

        if len(self.p2_mcts_wr_stats[0]) > 1:
            plt.plot(range(len(self.p2_mcts_wr_stats[0])), self.p2_mcts_wr_stats[0], label = "P1")
            plt.plot(range(len(self.p2_mcts_wr_stats[1])), self.p2_mcts_wr_stats[1], label = "P2")
            plt.title("MCTS -> Win rates as Player 2")
            plt.legend()
            plt.savefig(self.plots_path + 'p2_mcts_wr.png')
            plt.clf()
        
        print("Wr plotting done.\n")

    def plot_weight(self):
        print("\nPlotting weights...")
                    
        plt.plot(range(len(self.weight_size_max)), self.weight_size_max)
        plt.title("Max Weight")
        plt.savefig(self.plots_path + 'weight_max.png')
        plt.clf()

        plt.plot(range(len(self.weight_size_min)), self.weight_size_min)
        plt.title("Min Weight")
        plt.savefig(self.plots_path + 'weight_min.png')
        plt.clf()

        plt.plot(range(len(self.weight_size_average)), self.weight_size_average)
        plt.title("Average Weight")
        plt.savefig(self.plots_path + 'weight_average.png')
        plt.clf()

        print("Weight plotting done.\n")

    def plot_state_set(self):
        red = (200/255, 0, 0)
        grey = (65/255, 65/255, 65/255)
        green = (45/255, 110/255, 10/255)

        if len(self.state_set_stats[0]) > 1:
            print("\nPlotting state set...")
            for i in range(len(self.state_set_stats)):
                    if i<=1:
                        color = red
                    elif i<=3:
                        color = grey
                    else:
                        color = green
                        
                    plt.plot(range(len(self.state_set_stats[i])), self.state_set_stats[i], color=color)

            plt.title("State Values")
            plt.savefig(self.plots_path + '_state_values.png')
            plt.clf()
            print("State plotting done\n")

    def update_wr_data(self, result, result_type):
        # An integer between 0 and 3 is used to determine which list to update
        for player in (0,1):
            if result_type == 0:
                self.p1_policy_wr_stats[player].append(result[player])

            if result_type == 1:   
                self.p2_policy_wr_stats[player].append(result[player])

            if result_type == 2:
                self.p1_mcts_wr_stats[player].append(result[player])

            if result_type == 3:
                self.p2_mcts_wr_stats[player].append(result[player])
        return

    def update_weight_data(self):
        model = self.latest_network.get_model()
        all_weights = torch.Tensor().cpu()
        for param in model.parameters():
            all_weights = torch.cat((all_weights, param.clone().detach().flatten().cpu()), 0)

        self.weight_size_max.append(max(abs(all_weights)))
        self.weight_size_min.append(min(abs(all_weights)))
        self.weight_size_average.append(torch.mean(abs(all_weights)))
        del all_weights

    def update_state_set_data(self, test_iterations):
        for i in range(len(self.state_set)):
            state = self.state_set[i]
            _, value = self.latest_network.inference(state, False, test_iterations)
            self.state_set_stats[i].append(value.item())
    
    def split_policy_loss_graph(self):
        print("\nSpliting policy loss graph...")
        num_points = len(self.train_global_policy_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_policy_loss)
            plt.title("Before split global policy loss")
            plt.savefig(self.plots_path + "_" + self.network_name + '_before_split_global_policy_loss.png')
            plt.clf()

        self.train_global_policy_loss.clear()
        print("Spliting done.\n")
        return
    
    def split_value_loss_graph(self):
        print("\nSpliting value loss graph...")
        num_points = len(self.train_global_value_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_value_loss)
            plt.title("Before split global value loss")
            plt.savefig(self.plots_path + "_" + self.network_name + '_before_split_global_value_loss.png')
            plt.clf()

        self.train_global_value_loss.clear()
        print("Spliting done.")
        return
    
    def split_combined_loss_graph(self):
        print("\nSpliting cobmined loss graph...")
        num_points = len(self.train_global_combined_loss)
        if num_points > 1:
            x = range(num_points)
            plt.plot(x, self.train_global_combined_loss)
            plt.title("Before split global value loss")
            plt.savefig(self.plots_path + "_" + self.network_name + '_before_split_global_value_loss.png')
            plt.clf()

        self.train_global_combined_loss.clear()
        print("Spliting done.")
        return

    def save_data(self, data_path):
        # Save ploting information to use when continuing training
        with open(data_path, 'wb') as file:
            pickle.dump(self.epochs_value_loss, file)
            pickle.dump(self.epochs_policy_loss, file)
            pickle.dump(self.epochs_combined_loss, file)

            pickle.dump(self.train_global_value_loss, file)
            pickle.dump(self.train_global_policy_loss, file)
            pickle.dump(self.train_global_combined_loss, file)

            pickle.dump(self.weight_size_max, file)
            pickle.dump(self.weight_size_min, file)
            pickle.dump(self.weight_size_average, file)

            pickle.dump(self.p1_policy_wr_stats, file)
            pickle.dump(self.p2_policy_wr_stats, file)
            pickle.dump(self.p1_mcts_wr_stats, file)
            pickle.dump(self.p2_mcts_wr_stats, file)

            if self.state_set is not None:
                pickle.dump(self.state_set_stats, file)

    def load_data(self, data_path):
        # Load all the plot data
        with open(data_path, 'rb') as file:
            self.epochs_value_loss = pickle.load(file)
            self.epochs_policy_loss = pickle.load(file)
            self.epochs_combined_loss = pickle.load(file)

            self.train_global_value_loss = pickle.load(file)
            self.train_global_policy_loss = pickle.load(file)
            self.train_global_combined_loss = pickle.load(file)

            self.weight_size_max = pickle.load(file)
            self.weight_size_min = pickle.load(file)
            self.weight_size_average = pickle.load(file)

            self.p1_policy_wr_stats = pickle.load(file)
            self.p2_policy_wr_stats = pickle.load(file)
            self.p1_mcts_wr_stats = pickle.load(file)
            self.p2_mcts_wr_stats = pickle.load(file)

            if self.state_set is not None:
                self.state_set_stats = pickle.load(file)
    