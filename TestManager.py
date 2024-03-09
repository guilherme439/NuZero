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

from Agents.Generic.RandomAgent import RandomAgent
from Agents.Generic.PolicyAgent import PolicyAgent
from Agents.Generic.MctsAgent import MctsAgent
from Agents.SCS.GoalRushAgent import GoalRushAgent


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
    
    def run_test_batch(self, num_games, p1_agent, p2_agent, p1_keep_updated, p2_keep_updated, show_info=True):
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
        
            first_flags = [True, True]
            keep_updated_vars = [p1_keep_updated, p2_keep_updated]
            latest_caches = [None, None]
            current_caches = [p1_cache, p2_cache]
            for p in [0,1]:
                if keep_updated_vars[p]:
                    # The first game to finish initializes the cache
                    if first_flags[p]:   
                        latest_caches[p] = current_caches[p]
                        first_flags[p] = False
                    # The remaining games update the cache with the states they saw
                    else:       
                        # latest_cache could be None if the cache is disabled
                        if (latest_caches[p].get_fill_ratio() < latest_caches[p].get_update_threshold()):
                            latest_caches[p].update(current_caches[p])
                    
            
            # While there are games to play... we request more
            if games_requested < num_games:
                game = self.game_class(*self.game_args)

                call_args = [game, p1_agent, p2_agent, latest_caches[0], latest_caches[1], False]
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
    
    def test_from_config(self, test_config_path, model_p1=None, model_p2=None, show_info=True):
        test_config = load_yaml_config(test_config_path)
        new_testers = test_config["Testers"]["new_testers"]
        if new_testers:
            num_testers = test_config["Testers"]["num_testers"]
            slow = test_config["Testers"]["slow"]
            use_print = test_config["Testers"]["print"]
            render_choice = test_config["Testers"]["render_choice"]
            self.new_testers(num_testers, slow, use_print, render_choice)

        changing_agent = test_config["Test"]["changing_agent"]
        changing_parameter = test_config["Test"]["changing_parameter"]
        parameter_name = changing_parameter["name"]
        range_start = changing_parameter["Range"]["first"]
        range_end = changing_parameter["Range"]["last"] + 1
        range_step = changing_parameter["Range"]["step"]
        parameter_range = range(range_start, range_end, range_step)

        num_runs = test_config["Test"]["num_runs"]
        num_games_per_run = test_config["Test"]["num_games_per_run"]

        p1_agent_config = test_config["Agents"]["p1_agent"]
        p2_agent_config = test_config["Agents"]["p2_agent"]

        if changing_agent == 1:
            agent_to_change = p1_agent
            parameter_config = p1_agent_config
        elif changing_agent == 2:
            agent_to_change = p2_agent
            parameter_config = p2_agent_config

        if (changing_parameter == "checkpoints"):
            if not parameter_config["load_checkpoints"]:
                raise Exception("It is only possible to change network checkpoits if the agent is set to load them.")
            else:
                cp_network_name = parameter_config["Checkpoints"]["cp_network_name"]
            
        if model_p1:
            p1_agent, p1_keep_updated = self.create_agent_from_config(p1_agent_config, model_p1)
        else:
            p1_agent, p1_keep_updated = self.create_agent_from_config(p1_agent_config)

        if model_p2:
            p2_agent, p2_keep_updated = self.create_agent_from_config(p2_agent_config, model_p2)
        else:
            p2_agent, p2_keep_updated = self.create_agent_from_config(p2_agent_config)


        all_data = []
        for value in parameter_range:
            p1_avg = 0
            p2_avg = 0
            draw_avg = 0
            if parameter_name == "checkpoints":
                nn, _, _, _ = load_network_checkpoint(cp_network_name, value)
                agent_to_change.set_network(nn)
                print("-------------------\n\nCheckpoint: " + str(value))
            elif parameter_name == "iterations":
                agent_to_change.set_iterations(value)
                print("-------------------\n\nIteration: " + str(value))
                
            for run in range(num_runs):
                print("\nRun " + str(run))
                (p1_wr, p2_wr, draws) = self.run_test_batch(num_games_per_run, p1_agent, p2_agent, p1_keep_updated, p2_keep_updated, show_info)
                p1_avg += p1_wr/num_runs
                p2_avg += p2_wr/num_runs
                draw_avg += draws/num_runs
            
            wr_data = (p1_avg, p2_avg, draw_avg)
            data_point = (value, wr_data)
            all_data.append(data_point)

        return all_data
 
    def create_agent_from_config(self, agent_config, model=None):
        agent_type = agent_config["agent_type"]
        keep_updated = False
        if agent_type == "mcts" or agent_type == "policy":
            cache_config = agent_config["Cache"]
            cache_choice = cache_config["cache_choice"]
            if cache_choice != "disabled":
                max_size = cache_config["max_size"]
                keep_updated = cache_config["keep_updated"]
                cache = create_cache(cache_choice, max_size)
            else:
                cache = None

            network_config = agent_config["Network"]
            recurrent_iterations = network_config["recurrent_iterations"]
            load_checkpoint = network_config["load_checkpoint"]
            if load_checkpoint:
                cp_config = network_config["Checkpoints"]
                cp_network_name = cp_config["cp_network_name"]
                cp_number = cp_config["cp_number"]
                nn = load_network_checkpoint(cp_network_name, cp_number)
            else:
                nn = Torch_NN(model)

            if agent_type == "mcts":
                search_config_path = agent_config["search_config_path"]
                search_config = load_yaml_config(search_config_path)
                agent = MctsAgent(search_config, nn, recurrent_iterations, cache)

            elif agent_type == "policy":
                agent = PolicyAgent(nn, recurrent_iterations, cache)
            

        elif agent_type == "goal_rush":
            agent = GoalRushAgent()
        
        elif agent_type == "random":
            agent = RandomAgent()
        
        else:
            raise Exception("Agent type not recognized. Options: mcts, policy, goal_rush, random")
        
        return agent, keep_updated


    

    