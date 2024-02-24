import os
import ray
import psutil
import resource
import time
import torch
import pickle
import math
import copy
import numpy as np
import matplotlib.pyplot as plt


from torch import nn

from Neural_Networks.Torch_NN import Torch_NN

from ReplayBuffer import ReplayBuffer
from RemoteStorage import RemoteStorage
from RemoteTester import RemoteTester

from Utils.stats_utilities import *
from Utils.loss_functions import *
from Utils.other_utils import *
from Utils.PrintBar import PrintBar

from progress.bar import ChargingBar
from progress.spinner import PieSpinner


class TestManager():
    '''Runs tests and returns results'''
	
    def __init__(self, game_class, game_args, num_actors, shared_storage, keep_updated, cache_choice, cache_max_size):
        self.game_class = game_class
        self.game_args = game_args
        self.num_actors = num_actors
        self.shared_storage = shared_storage
        self.keep_updated = keep_updated
        self.cache_choice = cache_choice
        self.cache_max_size = cache_max_size

        # ------------------------------------------------------ #
        # --------------------- ACTOR POOL --------------------- #
        # ------------------------------------------------------ #

        
        actor_list = [RemoteTester.remote() for a in range(self.num_actors)]
        self.actor_pool = ray.util.ActorPool(actor_list)

    def run_group_of_test_batches(self, num_batches, games_per_batch, p1_agents_list, p2_agents_list, show_results):

        result_list = []
        for batch_idx in range(num_batches):
            p1_agent = p1_agents_list[batch_idx]
            p2_agent = p2_agents_list[batch_idx]
            result = self.run_test_batch(games_per_batch, p1_agent, p2_agent, show_results)
            result_list.append(result)

        return result_list
        

    def run_test_batch(self, num_games, p1_agent, p2_agent, show_results=True):
        start = time.time()
        print("\n")

        wins = [0,0]

        '''
        args = [None, p1_agent, p2_agent, False]


        # We must use actor_pool.map instead of actor_pool.submit,
        # because ray bugs if you do several submit calls on the same actors with different values
        map_args = []
        for g in range(num_games):
            game = self.game_class(*self.game_args)
            args_copy = copy.copy(args)
            args_copy[0] = game
            map_args.append(args_copy)

        results = self.actor_pool.map_unordered(lambda actor, args: actor.Test_using_agents.remote(*args), map_args)
            
        time.sleep(1)

        bar = PrintBar('Testing', num_games, 15)
        for res in results:
            winner, _ = res
            if winner != 0:
                wins[winner-1] +=1
            bar.next()

        bar.finish()

        '''
        bar = PrintBar('Testing', num_games, 15)

        call_args = [None, p1_agent, p2_agent]

        first_requests = min(self.num_actors, num_games)
        for r in range(first_requests):
            game = self.game_class(*self.game_args)
            call_args[0] = game
            self.actor_pool.submit(lambda actor, args: actor.Test_using_agents.remote(*args), call_args)

        first = True
        games_played = 0
        games_requested = first_requests
        while games_played < num_games:
        
            winner, _, p1_cache, p2_cache = self.actor_pool.get_next_unordered()
            games_played += 1
            bar.next()
            if winner != 0:
                wins[winner-1] +=1

            if self.keep_updated:
                if first:   
                    # The first game to finish initializes the cache
                    p1_latest_cache = p1_cache
                    p2_latest_cache = p2_cache
                    first = False
                else:       
                    # The remaining games update the cache with the states they saw
                    if p1_latest_cache.get_fill_ratio() < 0.7:
                        p1_latest_cache.update(p1_cache)
                    if p2_latest_cache.get_fill_ratio() < 0.7:
                        p2_latest_cache.update(p1_cache)
            
            # While there are games to play... we request more
            if games_requested < num_games:
                if self.keep_updated:
                    call_args = [None, p1_agent, p2_agent, p1_latest_cache, p2_latest_cache, False]

                game = self.game_class(*self.game_args)
                call_args[0] = game
                self.actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), call_args)
                games_requested +=1

        
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


        if show_results:
            print("\n\n")
            print("Results for:\n" + "p1->" + p1_agent.name() + "\np2->" + p2_agent.name() + "\n")
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
    

    def run_mcts_batch_with_stats(self):
        return
        

    
    

    