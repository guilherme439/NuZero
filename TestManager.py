import os
import ray
import psutil
import resource
import time
import torch
import pickle
import math
import numpy as np
import matplotlib.pyplot as plt


from copy import deepcopy
from torch import nn

from Neural_Networks.Torch_NN import Torch_NN

from ReplayBuffer import ReplayBuffer
from RemoteStorage import RemoteStorage
from RemoteTester import RemoteTester

from Utils.stats_utilities import *
from Utils.loss_functions import *
from Utils.PrintBar import PrintBar

from progress.bar import ChargingBar
from progress.spinner import PieSpinner


class TestManager():
    '''Runs tests and returns results'''
	
    def __init__(self, game_class, game_args, train_config, search_config, shared_storage, state_set=None):
        self.game_class = game_class
        self.game_args = game_args
        self.train_config = train_config
        self.search_config = search_config
        self.shared_storage = shared_storage
        self.state_set = state_set

        # ------------------------------------------------------ #
        # --------------------- ACTOR POOL --------------------- #
        # ------------------------------------------------------ #

        num_testers = self.train_config.testing["testing_actors"]
        actor_list = [RemoteTester.remote() for a in range(num_testers)]
        self.actor_pool = ray.util.ActorPool(actor_list)

    
    def run_tests(self, policy_games, mcts_games, state_cache):
        p1_policy_results = ()
        p2_policy_results = ()
        p1_mcts_results = ()
        p2_mcts_results = ()

        if policy_games:
            p1_policy_results = self.test_latest_nn("1", policy_games, "policy", state_cache)
            p2_policy_results = self.test_latest_nn("2", policy_games, "policy", state_cache) 

        if mcts_games:
            p1_mcts_results = self.test_latest_nn("1", mcts_games, "mcts", state_cache)
            p2_mcts_results = self.test_latest_nn("2", mcts_games, "mcts", state_cache)
            
        return p1_policy_results, p2_policy_results, p1_mcts_results, p2_mcts_results
       
    def test_latest_nn(self, player_choice, num_games, test_mode, state_cache, show_results=True):
        start = time.time()
        print("\n")

        latest_network = ray.get(self.shared_storage.get.remote()) # ask for a copy of the latest network

        test_iterations = self.train_config.recurrent_networks["num_test_iterations"]

        stats_list = []
        wins = [0,0]

        use_state_cache = False
        if state_cache != "disabled":
            use_state_cache = True

        if test_mode == "policy":
            args_list = [player_choice, None, latest_network, None, test_iterations, False]
            game_index = 1
        elif test_mode == "mcts":
            args_list = [player_choice, self.search_config, None, latest_network, None, test_iterations, use_state_cache, False]
            game_index = 2

        if show_results:
            print("\n\nTesting as p" + player_choice + " using " + test_mode)

        for g in range(num_games):
            game = self.game_class(*self.game_args)
            args_list[game_index] = game
            if test_mode == "policy":
                self.actor_pool.submit(lambda actor, args: actor.Test_AI_with_policy.remote(*args), args_list)
            elif test_mode == "mcts":
                self.actor_pool.submit(lambda actor, args: actor.Test_AI_with_mcts.remote(*args), args_list)

        time.sleep(5) # Sometimes ray bugs if we dont wait before getting the results

        for g in range(num_games):
            winner, stats = self.actor_pool.get_next_unordered() # Timeout and Ignore_if_timeout
            stats_list.append(stats)
            if winner != 0:
                wins[winner-1] += 1

        if test_mode == "mcts" and show_results:
            print_stats_list(stats_list)
        
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


        if show_results:
            print("\n\nAI playing as p" + player_choice + "\n")
            print("P1 Win ratio: " + format(p1_winrate, '.4'))
            print("P2 Win ratio: " + format(p2_winrate, '.4'))
            print("Draw percentage: " + format(draw_percentage, '.4'))
            print("Comparative Win ratio(p1/p2): " + cmp_1_string)
            print("Comparative Win ratio(p2/p1): " + cmp_2_string + "\n", flush=True)


        end = time.time()
        total_time = end-start
        print("\n\nTotal testing time(m): " + format(total_time/60, '.4'))
        print("Average time per game(s): " + format(total_time/num_games, '.4'))
        print("\n\n")

        return (p1_winrate, p2_winrate, draw_percentage)
        

    
    

    