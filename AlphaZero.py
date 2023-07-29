import math
import numpy as np
import os
import psutil
import gc
import resource
import random
import pickle
import time
import sys
import ray

import torch

from stats_utilities import *

from copy import deepcopy

from Gamer import Gamer
from Replay_Buffer import Replay_Buffer
from Shared_network_storage import Shared_network_storage

import matplotlib.pyplot as plt

from progress.bar import ChargingBar
from progress.spinner import PieSpinner
from termcolor import colored


from Neural_Networks.Torch_NN import Torch_NN

from RemoteTester import RemoteTester

from ray.runtime_env import RuntimeEnv

class AlphaZero():

	
	def __init__(self, model, recurrent, game_class, game_args, alpha_config, search_config, network_name="ABC", continuing=False):
		self.game_args = game_args  # Args for the game's __init__()
		self.game_class = game_class
		game = game_class(*game_args)

		self.network_name = network_name
		self.latest_network = Torch_NN(game, model, recurrent)
		self.model_class = model.__class__

		self.alpha_config = alpha_config
		self.search_config = search_config
		self.continuing = continuing

		self.n_updates = 0
		self.decisive_count = 0
		

		#Plots
		self.plot_loss = True
		self.plot_wr = True
		
		self.value_loss = []
		self.policy_loss = []
		self.total_loss = []

		self.epochs_value_loss = []
		self.epochs_policy_loss = []
		self.epochs_combined_loss = []

		self.tests_value_loss = []
		self.tests_policy_loss = []
		self.tests_combined_loss = []

		self.train_global_value_loss = []
		self.train_global_policy_loss = []
		self.train_global_combined_loss = []

		self.test_global_value_loss = []
		self.test_global_policy_loss = []
		self.test_global_combined_loss = []

		self.wr1_stats = [0.0]
		self.wr2_stats = [0.0]

		self.weight_size_max = []
		self.weight_size_min = []
		self.weight_size_average = []

	def run(self, starting_iteration=0):
		pid = os.getpid()
		process = psutil.Process(pid)

		# ------------------------------------------------------ #
        # ----------------- RAY INITIALIZATION ----------------- #
        # ------------------------------------------------------ #

		print("\n\n--------------------------------\n")
		runtime_env=RuntimeEnv \
					(
					conda="tese",
					working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
					env_vars=
							{
							"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6",
							}
					)
		
		context = ray.init(address="auto", runtime_env=runtime_env, log_to_driver=True)

		# ------------------------------------------------------ #
        # ------------------ RUNTIME CONFIGS ------------------- #
        # ------------------------------------------------------ #

		#NOTE: currently self-play uses the storage_network and testing uses the latest_network, they are only the same if storage_frequency=1

		state_cache = self.alpha_config.optimization["state_cache"]

		num_games_per_batch = self.alpha_config.running["num_games_per_batch"]
		num_batches = self.alpha_config.running["num_batches"]
		early_fill = self.alpha_config.running["early_fill"]
		num_wr_testing_games = self.alpha_config.running["num_wr_testing_games"]

		# Test set requires playing extra games
		test_set = self.alpha_config.running["test_set"]
		num_test_set_games = self.alpha_config.running["num_test_set_games"]

		save_frequency = self.alpha_config.frequency["save_frequency"]
		test_frequency = self.alpha_config.frequency["test_frequency"]
		debug_frequency = self.alpha_config.frequency["debug_frequency"]
		storage_frequency = self.alpha_config.frequency["storage_frequency"]
		plot_frequency = self.alpha_config.frequency["plot_frequency"]
		plot_reset = self.alpha_config.frequency["plot_reset"]

		if test_frequency % save_frequency != 0:
			print("\nInvalid values for save and/or test frequency.\nThe \"test_frequency\" value must be divisible by the \"save_frequency\" value.")
			return
		
		# ------------------------------------------------------ #
        # ------------------- FOLDERS SETUP -------------------- #
        # ------------------------------------------------------ #

		print("\n--------------------------------\n\n")
		self.game = self.game_class(*self.game_args)
		game_folder_name = self.game.get_name()

		model_folder_path = game_folder_name + "/models/" + self.network_name
		plots_path = model_folder_path + "/plots"
		if not os.path.exists(model_folder_path):
			os.mkdir(model_folder_path)
		if not os.path.exists(plots_path):
			os.mkdir(plots_path)
		
		# pickle the network class
		file_name = model_folder_path + "/base_model.pkl"
		with open(file_name, 'wb') as file:
			pickle.dump(self.latest_network.get_model(), file)
			print(f'Successfully pickled model class at "{file_name}".\n')

		# create copies of the config files
		print("\nCreating config file copies:")
		search_config_copy_path = model_folder_path + "/search_config_copy.ini"
		alpha_config_copy_path = model_folder_path + "/alpha_config_copy.ini"
		self.alpha_config.save(alpha_config_copy_path)
		self.search_config.save(search_config_copy_path)
		print("\n\n--------------------------------\n")

		# ------------------------------------------------------ #
        # ------------- STORAGE AND BUFFERS SETUP -------------- #
        # ------------------------------------------------------ #

		shared_storage_size = self.alpha_config.learning["shared_storage_size"]
		replay_window_size = self.alpha_config.learning["replay_window_size"]
		learning_method = self.alpha_config.learning["learning_method"]

		self.network_storage = Shared_network_storage.remote(shared_storage_size)
		future_save = self.network_storage.save_network.remote(self.latest_network)

		if learning_method == "epochs":
			batch_size = self.alpha_config.epochs["batch_size"]
			learning_epochs = self.alpha_config.epochs["learning_epochs"]
			plot_epoch = self.alpha_config.epochs["plot_epoch"]
		elif learning_method == "samples":
			batch_size = self.alpha_config.samples["batch_size"]

		self.replay_buffer = Replay_Buffer.remote(replay_window_size, batch_size)

		if test_set:
			batches_in_replay_buffer = replay_window_size / num_games_per_batch
			test_buffer_window = int(batches_in_replay_buffer * num_test_set_games)
			self.test_buffer = Replay_Buffer.remote(test_buffer_window, batch_size)
			# test_buffer_window is such that there is the same number of batches in both the replay and test buffer.
		else:
			self.test_buffer = None

		# ------------------------------------------------------ #
        # ------------------ OPTIMIZER SETUP ------------------- #
        # ------------------------------------------------------ #

		optimizer_name = self.alpha_config.optimizer["optimizer"]
		learning_rate = self.alpha_config.optimizer["learning_rate"]

		weight_decay = self.alpha_config.optimizer["weight_decay"]
		momentum = self.alpha_config.optimizer["momentum"]

		scheduler_boundaries = self.alpha_config.optimizer["scheduler_boundaries"]
		scheduler_gamma = self.alpha_config.optimizer["scheduler_gamma"]
		
		if optimizer_name == "Adam":
			optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=learning_rate)
		elif optimizer_name == "SGD":
			optimizer = torch.optim.SGD(self.latest_network.get_model().parameters(), lr=learning_rate, momentum=momentum, weight_decay = weight_decay)
		else:
			optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=learning_rate)
			print("Bad optimizer config.\nUsing Adam optimizer...")

		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_boundaries, gamma=scheduler_gamma)

		# ------------------------------------------------------ #
        # --------------------- ALPHAZERO ---------------------- #
        # ------------------------------------------------------ #

		print("\nRunning for " + str(num_batches) + " batches of " + str(num_games_per_batch) + " games each.")
		if state_cache != "disabled":
			print("\n-Using state dictonary as cache.")			  

		ray.get(future_save) # wait for the network to be in storage

		model = self.latest_network.get_model()
		model_dict = model.state_dict()

		if not self.continuing:
			# Initial save (untrained network)
			save_path = game_folder_name + "/models/" + self.network_name + "/" + self.network_name + "_0_model"
			torch.save(model_dict, save_path)

		# Weight graphs
		all_weights = torch.Tensor()
		for k, v in model_dict.items():
			print(k)
			if "weight" in k:
				all_weights = torch.cat((all_weights, v.flatten()))


		self.weight_size_max.append(max(abs(all_weights)))
		self.weight_size_min.append(min(abs(all_weights)))
		self.weight_size_average.append(torch.mean(abs(all_weights)))
		del all_weights
		
		updates = 0
		if early_fill > 0:
			print("\n\n\n\nEarly Buffer Fill")
			self.run_selfplay(early_fill, False, state_cache, text="Filling initial games")

		since_reset = 0
		batches_to_run = range(starting_iteration, num_batches)
		for b in batches_to_run:
			updated = True

			print("\n\n\n\nBatch: " + str(b+1))
			
			self.decisive_count = 0
			self.run_selfplay(num_games_per_batch, False, state_cache, text="Self-Play Games")

			print("\nreplay buffer:")	
			print(ray.get(self.replay_buffer.played_games.remote()))


			if test_set:
				self.decisive_count = 0
				self.run_selfplay(num_test_set_games, True, text="Test-set Games")
				print("\ntest buffer:")
				print(ray.get(self.test_buffer.played_games.remote()))
			
			

			print("\n\nLearning rate: " + str(scheduler.get_last_lr()[0]))
			self.train_network(optimizer, scheduler, batch_size, learning_method, test_set)

			if (((b+1) % storage_frequency) == 0):
				future_storage = self.network_storage.save_network.remote(self.latest_network)

			since_reset += 1
			updates +=1
			

			if (((b+1) % save_frequency) == 0):
				save_path = game_folder_name + "/models/" + self.network_name + "/" + self.network_name + "_" + str(b+1) + "_model"
				torch.save(self.latest_network.get_model().state_dict(), save_path)
			
			if (((b+1) % test_frequency) == 0) and updated:
				wr1, _, _ = self.run_tests("1", num_wr_testing_games, state_cache)
				_, wr2, _ = self.run_tests("2", num_wr_testing_games, state_cache)

				# save wr as p1 and p2 for plotting
				self.wr1_stats.append(wr1) 
				self.wr2_stats.append(wr2)

			if (((b+1) % debug_frequency) == 0):				
				pass			

			if (((b+1) % plot_frequency) == 0):
				
				model_dict = self.latest_network.get_model().state_dict()
				all_weights = torch.Tensor()
				for k, v in model_dict.items():
					if "weight" in k:
						all_weights = torch.cat((all_weights, v.flatten()))

				self.weight_size_max.append(max(abs(all_weights)))
				self.weight_size_min.append(min(abs(all_weights)))
				self.weight_size_average.append(torch.mean(abs(all_weights)))
				del all_weights

				plt.plot(range(since_reset+1), self.weight_size_max)
				plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_weight_max.png')
				plt.clf()

				plt.plot(range(since_reset+1), self.weight_size_min)
				plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_weight_min.png')
				plt.clf()

				plt.plot(range(since_reset+1), self.weight_size_average)
				plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_weight_average.png')
				plt.clf()


				if self.plot_wr:
					number_of_tests = range(0, math.floor(updates/test_frequency) + 1)
					plt.plot(number_of_tests, self.wr1_stats, label = "Wr as P1")
					plt.plot(number_of_tests, self.wr2_stats, label = "Wr as P2")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_wr.png')
					plt.clf()

				if self.plot_loss:

					if (learning_method == "epochs"):
						if plot_epoch and (learning_epochs > 1):

							plt.plot(range(learning_epochs), self.epochs_value_loss, label = "Training")
							if test_set:
								plt.plot(range(learning_epochs), self.tests_value_loss, label = "Testing")

							plt.legend()
							plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_value_loss_' + str((b+1)) + '.png')
							plt.clf()

							
							plt.plot(range(learning_epochs), self.epochs_policy_loss, label = "Training")
							if test_set:
								plt.plot(range(learning_epochs), self.tests_policy_loss, label = "Testing")

							plt.legend()
							plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_policy_loss_' + str((b+1)) + '.png')
							plt.clf()

							
							plt.plot(range(learning_epochs), self.epochs_combined_loss, label = "Training")
							if test_set:
								plt.plot(range(learning_epochs), self.tests_combined_loss, label = "Testing")

							plt.legend()
							plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_total_loss_' + str((b+1)) + '.png')
							plt.clf()

					if learning_method == "epochs":
						number_of_data_points = learning_epochs*(since_reset)
					else:
						number_of_data_points = since_reset

					x = range(number_of_data_points)
					plt.plot(x, self.train_global_value_loss, label = "Training")
					if test_set:
						plt.plot(x, self.test_global_value_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_value_loss.png')
					plt.clf()

					plt.plot(x, self.train_global_policy_loss, label = "Training")
					if test_set:
						plt.plot(x, self.test_global_policy_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_policy_loss.png')
					plt.clf()

					plt.plot(x, self.train_global_combined_loss, label = "Training")
					if test_set:
						plt.plot(x, self.test_global_combined_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_total_loss.png')
					plt.clf()
				
			if (((b+1) % plot_reset) == 0):
				self.policy_loss.clear()
				self.value_loss.clear()
				self.total_loss.clear()
				self.train_global_combined_loss.clear()
				self.train_global_policy_loss.clear()
				self.train_global_value_loss.clear()
				since_reset = 0
			
			print("\n\nMain process memory usage: ")
			print("Current memory usage: " + format(process.memory_info().rss/(1024*1000), '.6') + " MB") 
			print("Peak memory usage: " + format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1000, '.6') + " MB" )
			# psutil gives memory in bytes and resource gives memory in kb (1024 bytes)

			
			ray.get(future_storage) # wait for the network to be stored before next iteration	
			
		
		return
	
	def test_set_loss(self, batch, batch_size, iterations):

		combined_loss = 0.0
		policy_loss = 0.0
		value_loss = 0.0
		for (state, (target_value, target_policy)) in batch:
			
			predicted_policy, predicted_value = self.latest_network.inference(state, False, iterations)

			target_policy = torch.tensor(target_policy).to(self.latest_network.device)
			target_value = torch.tensor(target_value).to(self.latest_network.device)

			
			#policy_loss += ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) / math.log(len(target_policy)) )
			policy_loss += ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) )
			#Policy loss is being "normalized" by log(num_actions), since cross entropy's expected value is log(target_size)

			value_loss += ((target_value - predicted_value) ** 2)
			#value_loss += torch.abs(target_value - predicted_value)
			
		value_loss /= batch_size
		policy_loss /= batch_size

		combined_loss = policy_loss + value_loss

		return value_loss.item(), policy_loss.item(), combined_loss.item()

	def batch_update_weights(self, optimizer, scheduler, batch, batch_size, train_iterations):

		self.latest_network.get_model().train()
		optimizer.zero_grad()

		loss = 0.0
		policy_loss = 0.0
		value_loss = 0.0

		states, targets = list(zip(*batch))
		values, policies = list(zip(*targets))

		batch_input = torch.cat(states, 0)

		predicted_policies, predicted_values = self.latest_network.inference(batch_input, True, train_iterations)

		for i in range(batch_size):
			
			target_policy = torch.tensor(policies[i]).to(self.latest_network.device)
			target_value = torch.tensor(values[i]).to(self.latest_network.device)

			predicted_value = predicted_values[i]
			predicted_policy = predicted_policies[i]

			policy_loss += ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) / math.log(len(target_policy)) )
			#policy_loss += ( (-torch.sum(target_policy * torch.log(predicted_policy.flatten()))) )
			#Policy loss is being "normalized" by log(num_actions), since cross entropy's expected value is log(target_size)

			value_loss += ((target_value - predicted_value) ** 2)
			#value_loss += torch.abs(target_value - predicted_value)
			
		value_loss /= batch_size
		policy_loss /= batch_size

		loss = policy_loss + value_loss
		#loss = value_loss

		'''
		if self.plot_loss:
			self.policy_loss.append(policy_loss.item())
			self.value_loss.append(value_loss.item())
			self.total_loss.append(loss.item())
		'''

		#print(loss)
		# In PyThorch SGD optimizer already applies L2 weight regularization
		
		loss.backward()
		optimizer.step()
		scheduler.step()

		return value_loss.item(), policy_loss.item(), loss.item()

	def train_network(self, optimizer, scheduler, batch_size, learning_method, test_set):
		print()
		start = time.time()

		train_iterations = self.alpha_config.recurrent_networks["num_train_iterations"]

		replay_size = ray.get(self.replay_buffer.len.remote(), timeout=120)
		n_games = ray.get(self.replay_buffer.played_games.remote(), timeout=120)
		if test_set:
			test_size = ray.get(self.test_buffer.len.remote(), timeout=120)

		print("\nAfter Self-Play there are a total of " + str(replay_size) + " positions in the replay buffer.")
		print("Total number of games: " + str(n_games))
		
		if learning_method == "epochs":
			learning_epochs = self.alpha_config.epochs["learning_epochs"]
			
			if  batch_size > replay_size:
				print("Batch size too large.\n" + 
					"If you want to use batche_size with more moves than the first batch of games played " + 
					"you need to use, the \"early_fill\" config to fill the batch with random moves at the beggining.\n")
				exit()
			else:
				number_of_batches = replay_size // batch_size
				print("Batches in replay buffer: " + str(number_of_batches))

				if test_set:
					number_of_test_batches = test_size // batch_size
					print("Batches in test buffer: " + str(number_of_test_batches))

				print("Batch size: " + str(batch_size))	
				print("\n")

			value_loss = 0.0
			policy_loss = 0.0
			combined_loss = 0.0

			self.epochs_value_loss.clear()
			self.epochs_policy_loss.clear()
			self.epochs_combined_loss.clear()

			self.tests_value_loss.clear()
			self.tests_policy_loss.clear()
			self.tests_combined_loss.clear()

			
			total_updates = learning_epochs*number_of_batches
			print("\nTotal number of updates: " + str(total_updates) + "\n")
			
			bar = ChargingBar('Training ', max=learning_epochs)
			for e in range(learning_epochs):

				ray.get(self.replay_buffer.shuffle.remote(), timeout=120) # ray.get() beacuse we want the shuffle to be done before using buffer
				if test_set:
					future_test_shuffle = self.test_buffer.shuffle.remote()

				future_replay_buffer = self.replay_buffer.get_buffer.remote()
				if test_set:
					future_test_buffer = self.test_buffer.get_buffer.remote()

				epoch_value_loss = 0.0
				epoch_policy_loss = 0.0
				epoch_combined_loss = 0.0

				if test_set:
					t_epoch_value_loss = 0.0
					t_epoch_policy_loss = 0.0
					t_epoch_combined_loss = 0.0

				spinner = PieSpinner('\t\t\t\t\t\t  Running epoch ')
				# We get entire buffer and slice locally to avoid a lot of remote calls (buffer.get_slice could also be used)
				replay_buffer = ray.get(future_replay_buffer, timeout=300) 
				for b in range(number_of_batches):		
					start_index = b*batch_size
					next_index = (b+1)*batch_size

					batch = replay_buffer[start_index:next_index] # the slice does not take the last element
				
					value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler, batch, batch_size, train_iterations)

					epoch_value_loss += value_loss
					epoch_policy_loss += policy_loss
					epoch_combined_loss += combined_loss
					spinner.next()

				epoch_value_loss /= number_of_batches
				epoch_policy_loss /= number_of_batches
				epoch_combined_loss /= number_of_batches	
				
				if test_set:
					ray.get(future_test_shuffle)
					test_buffer = ray.get(future_test_buffer)
					for t in range(number_of_test_batches):
						start_index = t*batch_size
						next_index = (t+1)*batch_size

						batch = test_buffer[start_index:next_index]
					
						test_value_loss, test_policy_loss, test_combined_loss = self.test_set_loss(batch, batch_size, train_iterations)

						t_epoch_value_loss += test_value_loss
						t_epoch_policy_loss += test_policy_loss
						t_epoch_combined_loss += test_combined_loss
						spinner.next()
						
					t_epoch_value_loss /= number_of_test_batches
					t_epoch_policy_loss /= number_of_test_batches
					t_epoch_combined_loss /= number_of_test_batches	

				if self.plot_loss:
					self.epochs_value_loss.append(epoch_value_loss)
					self.epochs_policy_loss.append(epoch_policy_loss)
					self.epochs_combined_loss.append(epoch_combined_loss)

					if test_set:
						self.tests_value_loss.append(test_value_loss)
						self.tests_policy_loss.append(test_policy_loss)
						self.tests_combined_loss.append(test_combined_loss)

				spinner.finish
				bar.next()
				
			bar.finish()

			self.train_global_value_loss.extend(self.epochs_value_loss)
			self.train_global_policy_loss.extend(self.epochs_policy_loss)
			self.train_global_combined_loss.extend(self.epochs_combined_loss)

			if test_set:
				self.test_global_value_loss.extend(self.tests_value_loss)
				self.test_global_policy_loss.extend(self.tests_policy_loss)
				self.test_global_combined_loss.extend(self.tests_combined_loss)
				
		elif learning_method == "samples":
			num_samples = self.alpha_config.samples["num_samples"]
			late_heavy = self.alpha_config.samples["late_heavy"]

			future_buffer = self.replay_buffer.get_buffer.remote()
			batch = []
			probs = []
			if late_heavy:
				# The way I found to create a scalling array
				variation = 0.6 # number between 0 and 1
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
			bar = ChargingBar('Training ', max=num_samples)
			replay_buffer = ray.get(future_buffer, timeout=300)
			for _ in range(num_samples):
				if probs == []:
					batch_indexes = np.random.choice(len(replay_buffer), size=batch_size, replace=True)
				else:
					batch_indexes = np.random.choice(len(replay_buffer), size=batch_size, replace=True, p=probs)
				
				batch = [replay_buffer[i] for i in batch_indexes]

				value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler, batch, batch_size, train_iterations)

				average_value_loss += value_loss
				average_policy_loss += policy_loss
				average_combined_loss += combined_loss

				bar.next()

			bar.finish

			self.train_global_value_loss.append(average_value_loss/num_samples)
			self.train_global_policy_loss.append(average_policy_loss/num_samples)
			self.train_global_combined_loss.append(average_combined_loss/num_samples)

		else:
			print("Bad learning_method config.\nExiting")
			exit()

		
		end = time.time()
		total_time = end-start
		print("\n\nTraining time(s): " + format(total_time, '.4') + "\n")
		return
			
	def run_selfplay(self, num_games_per_batch, test_set, state_cache, text="Self-Play"):
		start = time.time()
		print("\n")

		pred_iterations = self.alpha_config.recurrent_networks["num_pred_iterations"]

		num_actors = self.alpha_config.actors["num_actors"]
		chunk_size = self.alpha_config.actors["chunk_size"]

		if test_set:
			buffer_to_use = self.test_buffer
		else:
			buffer_to_use = self.replay_buffer

		num_chunks = num_games_per_batch // chunk_size
		rest = num_games_per_batch % chunk_size

		stats_list = []
		args_list = []
		bar = ChargingBar(text, max=num_games_per_batch)
		bar.next(0)
		for c in range(num_chunks+1):
			games_to_play = chunk_size
			if c == num_chunks:
				games_to_play = rest

			actor_list= [Gamer.remote
		 				(
						buffer_to_use,
						self.network_storage,
						self.game_class,
						self.game_args,
						self.search_config,
						pred_iterations,
						state_cache
						)
		 				for a in range(num_actors)]
			
			actor_pool = ray.util.ActorPool(actor_list)

			for g in range(games_to_play):
				actor_pool.submit(lambda actor, args: actor.play_game.remote(*args), args_list)

			
			for g in range(games_to_play):
				stats = actor_pool.get_next_unordered(250, True) # Timeout and Ignore_if_timeout
				stats_list.append(stats)
				bar.next()
	
		bar.finish()
		print_stats_list(stats_list)

		end = time.time()
		total_time = end-start
		print("\n\nTotal time(m): " + format(total_time/60, '.4'))
		print("Average time per game(s): " + format(total_time/num_games_per_batch, '.4'))

		return
	
	def run_tests(self, player_choice, num_games, state_cache, show_results=True, text="Testing"):	
		start = time.time()
		print("\n")

		test_mode = self.alpha_config.running["testing_mode"]
		test_iterations = self.alpha_config.recurrent_networks["num_test_iterations"]
		num_actors = self.alpha_config.actors["num_actors"]
		chunk_size = self.alpha_config.actors["chunk_size"]

		stats_list = []
		wins = [0,0]
		num_chunks = num_games // chunk_size
		rest = num_games % chunk_size

		use_state_cache = False
		if state_cache != "disabled":
			use_state_cache = True
		
		if test_mode == "policy":
			args_list = [player_choice, None, self.latest_network, test_iterations]
		elif test_mode == "mcts":
			args_list = [player_choice, None, self.latest_network, self.search_config, use_state_cache, test_iterations]

		bar = ChargingBar(text, max=num_games)
		bar.next(0)
		for c in range(num_chunks+1):
			games_to_play = chunk_size
			if c == num_chunks:
				games_to_play = rest

			actor_list = [RemoteTester.remote() for a in range(num_actors)]
			actor_pool = ray.util.ActorPool(actor_list)

			for g in range(games_to_play):
				game = self.game_class(*self.game_args)
				args_list[1] = game
				if test_mode == "policy":
					actor_pool.submit(lambda actor, args: actor.Test_AI_with_policy.remote(*args), args_list)
				elif test_mode == "mcts":
					actor_pool.submit(lambda actor, args: actor.Test_AI_with_mcts.remote(*args), args_list)

				
			
			for g in range(games_to_play):
				winner, stats = actor_pool.get_next_unordered(250, True) # Timeout and Ignore_if_timeout
				stats_list.append(stats)
				if winner != 0:
					wins[winner-1] +=1
				bar.next()
			
		bar.finish()

		if test_mode == "mcts":
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

		return p1_winrate, p2_winrate, draw_percentage



	








