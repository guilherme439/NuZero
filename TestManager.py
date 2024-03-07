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


from Utils.loss_functions import *
from Utils.general_utils import *
from Utils.Progress_Bars.PrintBar import PrintBar

from progress.bar import ChargingBar
from progress.spinner import PieSpinner


class TestManager():
    '''Runs tests and returns result's data'''
	
    def __init__(self, game_class, game_args, num_actors=1, slow=False, print=False, render_choice="disabled"):
        self.game_class = game_class
        self.game_args = game_args

        self.new_testers(num_actors, slow, print, render_choice)

    def new_testers(self, num_actors, slow=False, print=False, render_choice="disabled"):
        self.num_actors = num_actors

        passive_render = False
        if render_choice == "passive":
            passive_render = True

        actor_list = [RemoteTester.remote(slow=slow, print=print, passive_render=passive_render) for a in range(self.num_actors)]
        self.actor_pool = ray.util.ActorPool(actor_list)

    def change_game(self, game_class, game_args):
        self.game_class = game_class
        self.game_args = game_args
    
    def run_test_batch(self, num_games, p1_agent, p2_agent, keep_updated, show_info=True):
        start = time.time()

        wins = [0,0]
        
        if show_info:
            print("\n")
            bar = PrintBar('Testing', num_games, 15)

        first_requests = min(self.num_actors, num_games)
        for r in range(first_requests):
            game = self.game_class(*self.game_args)
            call_args = [game, p1_agent, p2_agent]
            self.actor_pool.submit(lambda actor, args: actor.Test_using_agents.remote(*args), call_args)

        first = True
        games_played = 0
        games_requested = first_requests
        while games_played < num_games:
        
            winner, _, p1_cache, p2_cache = self.actor_pool.get_next_unordered()
            games_played += 1
            if show_info:
                bar.next()
            if winner != 0:
                wins[winner-1] +=1

            if keep_updated:
                # The first game to finish initializes the cache
                if first:   
                    p1_latest_cache = p1_cache
                    p2_latest_cache = p2_cache
                    first = False
                # The remaining games update the cache with the states they saw
                else:       
                    # latest_cache could be None if the cache is disabled
                    if (p1_latest_cache is not None) and (p1_latest_cache.get_fill_ratio() < p1_latest_cache.get_update_threshold()):
                        p1_latest_cache.update(p1_cache)
                    if (p2_latest_cache is not None) and (p2_latest_cache.get_fill_ratio() < p2_latest_cache.get_update_threshold()):
                        p2_latest_cache.update(p2_cache)
            
            # While there are games to play... we request more
            if games_requested < num_games:
                game = self.game_class(*self.game_args)
                if keep_updated:
                    call_args = [game, p1_agent, p2_agent, p1_latest_cache, p2_latest_cache, False]
                else:
                    call_args = [game, p1_agent, p2_agent]
                
                self.actor_pool.submit(lambda actor, args: actor.Test_using_agents.remote(*args), call_args)
                games_requested +=1

        if show_info:
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


        if show_info:
            print("\n\n")
            print("Results for:\n" + "p1->" + p1_agent.name() + "\np2->" + p2_agent.name() + "\n")
            print("P1 Win ratio: " + format(p1_winrate, '.4'))
            print("P2 Win ratio: " + format(p2_winrate, '.4'))
            print("Draw percentage: " + format(draw_percentage, '.4'))
            print("Comparative Win ratio(p1/p2): " + cmp_1_string)
            print("Comparative Win ratio(p2/p1): " + cmp_2_string + "\n", flush=True)


        end = time.time()
        total_time = end-start
        if show_info:
            print("\n\nTotal testing time(m): " + format(total_time/60, '.4'))
            print("Average time per game(s): " + format(total_time/num_games, '.4'))
            print("\n\n")

        return (p1_winrate, p2_winrate, draw_percentage)
    
    def test_from_config(self, test_config_path, show_info=True):
        test_config = load_yaml_config(test_config_path)
        new_testers = test_config["Testers"]["new_testers"]
        if new_testers:
            num_testers = test_config["Testers"]["num_testers"]
            slow = test_config["Testers"]["slow"]
            use_print = test_config["Testers"]["print"]
            render_choice = test_config["Testers"]["render_choice"]
            self.new_testers(num_testers, slow, use_print, render_choice)

        
        num_runs = test_config["Runs"]["num_runs"]
        num_games_per_run = test_config["Runs"]["num_games_per_run"]

        p1_agent_config = test_config["Agents"]["p1_agent"]
        p2_agent_config = test_config["Agents"]["p2_agent"]

        # Previously we were creating a new agent evertime we changed neural network or the number of recurrent iterations
        # we might need to find a better way of doing this
        # as always, be carefull with the cache

        #p1_agent = self.create_agent_from_config(p1_agent_config)
        #p2_agent = self.create_agent_from_config(p2_agent_config)

        for cp in checkpoint_range:
            for iter in iteration_range:
                for run in range(num_runs):
                    print("Run " + str(run))
                    for k in range(num_rec_iters):
                        rec_iter = recurrent_iterations_list[k]
                        p1_agent = RandomAgent()
                        p2_agent = PolicyAgent(nn, rec_iter)
                        print("\n\n\nTesting with " + str(rec_iter) + " iterations\n")
                        p1_wr, p2_wr, _ = test_manager.run_test_batch(num_games, p1_agent, p2_agent, True)
                        #p1_wr_list[i][k] += p1_wr/num_runs_per_game
                        p2_wr_list[i][k] += p2_wr/num_runs_per_game

    def create_agent_from_config(self, agent_config):
        agent_type = agent_config["agent_type"]
        if agent_type == "random":
            return RandomAgent()
        elif agent_type == "policy":
            nn = self.load_network(agent_config)
            rec_iter = agent_config["recurrent_iterations"]
            return PolicyAgent(nn, rec_iter)
        else:
            raise Exception("Bad agent type in config file")

    def load_network_checkpoint(self, network_name, iteration_number):
        game_folder = "Games/" + self.example_game.get_name() + "/"
        cp_network_folder = game_folder + "models/" + network_name + "/"
        if not os.path.exists(cp_network_folder):
            raise Exception("Could not find a model with that name.\n \
                    If you are using Ray jobs with a working_directory,\
                    only the models uploaded to git will be available.")
        
        self.buffer_load_path = cp_network_folder + "replay_buffer.cp"

        if iteration_number == "auto":
            cp_paths = glob.glob(cp_network_folder + "*_cp")
            # finds all numbers in string -> gets the last one -> converts to int -> orders the numbers -> gets last number
            iteration_number = sorted(list(map(lambda str: int(re.findall('\d+',  str)[-1]), cp_paths)))[-1]    

        checkpoint_path =  cp_network_folder + network_name + "_" + str(iteration_number) + "_cp"
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        model_pickle_path =  cp_network_folder + "base_model.pkl"
        model = self.load_pickle(model_pickle_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer_pickle_path =  cp_network_folder + "base_optimizer.pkl"
        base_optimizer = self.load_pickle(optimizer_pickle_path)
        optimizer_dict = checkpoint["optimizer_state_dict"]

        scheduler_pickle_path =  cp_network_folder + "base_scheduler.pkl"
        base_scheduler = self.load_pickle(scheduler_pickle_path)
        scheduler_dict = checkpoint["scheduler_state_dict"]

        if not self.fresh_start:
            self.starting_step = iteration_number
            cp_plot_data_path = cp_network_folder + "plot_data.pkl"
            self.load_plot_data(cp_plot_data_path, iteration_number-1)

        nn = Torch_NN(model)
        return nn, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict
    
    

    