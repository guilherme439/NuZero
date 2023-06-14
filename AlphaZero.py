import math
import numpy as np
import os
import psutil
import gc
import time
import sys
import pickle
import ray
import random

from copy import deepcopy

from Gamer import Gamer
from Replay_Buffer import Replay_Buffer
from Shared_network_storage import Shared_network_storage

import matplotlib.pyplot as plt

from progress.bar import ChargingBar
from progress.spinner import PieSpinner
from termcolor import colored

import torch
from torch import nn
from scipy.special import softmax

from Node import Node

from Neural_Networks.Torch_NN import Torch_NN

from Tester import Tester
from RemoteTester import RemoteTester

from ray.runtime_env import RuntimeEnv

class AlphaZero():

	
	def __init__(self, model, recurrent, game_class, game_args, config, network_name="ABC"):
		self.network_name = network_name
		self.latest_network = Torch_NN(model, recurrent)
		self.model_class = model.__class__

		self.game_args = game_args  # Args for the game's __init__()
		self.game_class = game_class
		self.config = config


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

		# Performance
		self.dict_cache = False
		
		self.num_games_per_batch = 0

		self.state_values = {}


	def run(self):
		pid = os.getpid()
		process = psutil.Process(pid)

		
		runtime_env = RuntimeEnv(
								conda="tese",
								working_dir="https://github.com/guilherme439/NuZero/archive/refs/heads/main.zip",
								env_vars={"LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"}
								)
		
		context = ray.init(runtime_env=runtime_env, log_to_driver=False)
		print(context.dashboard_url)

		print("\n\nTO DOs: \
			\n- PARALELIZE TRAINING \
			\n- RENDERING / BOARD VISUALIZATION \
			\n- CONTINUOUS TRAINING \
			\n- MAYBE ASYNC SELF-PLAY AND TESTING\n\n")


		if self.config.test_frequency % self.config.save_frequency != 0:
			print("Invalid values for save and/or test frequency.\nThe \"test_frequency\" value must be divisible by the \"save_frequency\" value.")
			return
		
		self.num_games_per_batch = self.config.num_games_per_batch
		self.game = self.game_class(*self.game_args)
		self.dict_cache = self.config.with_cache
		game_folder_name = self.game.get_name()


		model_folder_path = game_folder_name + "/models/" + self.network_name
		plots_path = model_folder_path + "/plots"
		if not os.path.exists(model_folder_path):
			os.mkdir(model_folder_path)
		if not os.path.exists(plots_path):
			os.mkdir(plots_path)
		
		file_name = model_folder_path + "/Network.pkl"
		with open(file_name, 'wb') as file:
			pickle.dump(self.latest_network, file)
			print(f'Successfully saved network at "{file_name}".\n')


		print("\n|Running for " + str(self.config.num_batches) + " batches of " + str(self.config.num_games_per_batch) + " games each." )
			
		if self.dict_cache:
			print("\n-Using state dictonary as cache.")
		

		total_num_batches = self.config.replay_window_size / self.config.num_games_per_batch
		test_buffer_window = int(total_num_batches * self.config.num_test_set_games)

		self.replay_buffer = Replay_Buffer.remote(self.config.replay_window_size, self.config.batch_size)

		if self.config.test_set:
			self.test_buffer = Replay_Buffer.remote(test_buffer_window, self.config.batch_size)
		else:
			self.test_buffer = None

		
		if self.config.optimizer == "Adam":
			optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=self.config.learning_rate)
		elif self.config.optimizer == "SGD":
			optimizer = torch.optim.SGD(self.latest_network.get_model().parameters(), lr=self.config.learning_rate, momentum=self.config.momentum, weight_decay = self.config.weight_decay)
		else:
			optimizer = torch.optim.Adam(self.latest_network.get_model().parameters(), lr=self.config.learning_rate)
			print("Bad optimizer config.\nUsing Adam optimizer...")

		scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config.scheduler_boundaries, gamma=self.config.scheduler_gama)

		self.shared_storage = Shared_network_storage.remote(self.config.shared_storage_size)


		save_path = game_folder_name + "/models/" + self.network_name + "/" + self.network_name + "_0_model"
		torch.save(self.latest_network.get_model().state_dict(), save_path)

		ray.get(self.shared_storage.save_network.remote(self.latest_network))

		updates = 0
		if self.config.early_fill > 0:
			self.run_selfplay(self.config.early_fill, False, False, text="Filling initial games")


		since_reset = 0
		for b in range(self.config.num_batches):

			self.average_value_loss = 0
			updated = True
			print("\n")
			print("\nBatch: " + str(b+1))
			print("\n")
			
			self.decisive_count = 0
			self.run_selfplay(self.config.num_games_per_batch, False, False, text="Self-Play Games")

			print("\nreplay buffer:")	
			print(ray.get(self.replay_buffer.played_games.remote()))

			if self.config.test_set:
				self.decisive_count = 0
				self.run_selfplay(self.config.num_test_set_games, True, False, text="Test-set Games")
				print("\ntest buffer:")
				print(ray.get(self.test_buffer.played_games.remote()))
			
			

			print("\n\nLearning rate: " + str(scheduler.get_last_lr()[0]))
			self.train_network(optimizer, scheduler)

			if (((b+1) % self.config.storage_frequency) == 0):
				ray.get(self.shared_storage.save_network.remote(self.latest_network)) # Should delay ray.get() calls

			since_reset += 1
			updates +=1
			

			if (((b+1) % self.config.save_frequency) == 0):
				save_path = game_folder_name + "/models/" + self.network_name + "/" + self.network_name + "_" + str(b+1) + "_model"
				torch.save(self.latest_network.get_model().state_dict(), save_path)
			
			if (((b+1) % self.config.test_frequency) == 0) and updated:
				wr1, _, _ = self.run_tests(1, self.config.num_wr_testing_games)
				_, wr2, _ = self.run_tests(2, self.config.num_wr_testing_games)

				# save wr as p1 and p2 for plotting
				self.wr1_stats.append(wr1) 
				self.wr2_stats.append(wr2)

			if (((b+1) % self.config.debug_frequency) == 0):				
				pass
				#debug_tester = Tester.remote(recurrent_iters=self.config.num_testing_iters, debug=True)
				#_, _, _, state_history = ray.get(debug_tester.Test_AI_vs_AI_with_policy.remote(self.game_class, self.game_args, self.latest_network, self.latest_network), timeout=30)				

			if (((b+1) % self.config.plot_frequency) == 0):
				if self.plot_wr:
					number_of_tests = range(1, math.floor(updates/self.config.test_frequency) + 1+1)
					plt.plot(number_of_tests, self.wr1_stats, label = "Wr as P1")
					plt.plot(number_of_tests, self.wr2_stats, label = "Wr as P2")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_wr.png')
					plt.clf()

				if self.plot_loss:
					
					if (self.config.learning_method == "epochs") and (self.config.learning_epochs > 1):
						plt.plot(range(self.config.learning_epochs), self.epochs_value_loss, label = "Training")
						if self.config.test_set:
							plt.plot(range(self.config.learning_epochs), self.tests_value_loss, label = "Testing")

						plt.legend()
						plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_value_loss_' + str((b+1)) + '.png')
						plt.clf()

						
						plt.plot(range(self.config.learning_epochs), self.epochs_policy_loss, label = "Training")
						if self.config.test_set:
							plt.plot(range(self.config.learning_epochs), self.tests_policy_loss, label = "Testing")

						plt.legend()
						plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_policy_loss_' + str((b+1)) + '.png')
						plt.clf()

						
						plt.plot(range(self.config.learning_epochs), self.epochs_combined_loss, label = "Training")
						if self.config.test_set:
							plt.plot(range(self.config.learning_epochs), self.tests_combined_loss, label = "Testing")

						plt.legend()
						plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/" + self.network_name + '_total_loss_' + str((b+1)) + '.png')
						plt.clf()

					if self.config.learning_method == "epochs":
						number_of_data_points = self.config.learning_epochs*(since_reset)
					else:
						number_of_data_points = since_reset

					x = range(number_of_data_points)
					plt.plot(x, self.train_global_value_loss, label = "Training")
					if self.config.test_set:
						plt.plot(x, self.test_global_value_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_value_loss.png')
					plt.clf()

					plt.plot(x, self.train_global_policy_loss, label = "Training")
					if self.config.test_set:
						plt.plot(x, self.test_global_policy_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_policy_loss.png')
					plt.clf()

					plt.plot(x, self.train_global_combined_loss, label = "Training")
					if self.config.test_set:
						plt.plot(x, self.test_global_combined_loss, label = "Testing")

					plt.legend()
					plt.savefig(game_folder_name + "/models/" + self.network_name + "/plots/_" + self.network_name + '_global_total_loss.png')
					plt.clf()
				
			if (((b+1) % self.config.plot_reset) == 0):
				self.policy_loss.clear()
				self.value_loss.clear()
				self.total_loss.clear()
				self.train_global_combined_loss.clear()
				self.train_global_policy_loss.clear()
				self.train_global_value_loss.clear()
				since_reset = 0
										
		
		if self.config.plot:
			if self.plot_wr:
				number_of_tests = range(1, math.floor(updates/self.config.test_frequency) + 1+1)
				plt.plot(number_of_tests, self.wr1_stats, label = "Wr as P1")
				plt.plot(number_of_tests, self.wr2_stats, label = "Wr as P2")
				plt.legend()
				plt.show()
			if self.plot_loss:
				plt.plot(self.total_loss)
				plt.show()

		cfg_path = game_folder_name + "/models/" + self.network_name + "/" + str(self.network_name) + ".cfg"
		self.config.save_config(cfg_path, self.network_name, self.config.num_batches, self.config.num_games_per_batch)


		return
	
	def test_set_loss(self, batch, batch_size):

		combined_loss = 0.0
		policy_loss = 0.0
		value_loss = 0.0
		for (state, (target_value, target_policy)) in batch:
			
			predicted_policy, predicted_value = self.latest_network.inference(state, False, self.config.num_training_iters)

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

	def update_weights(self, optimizer, scheduler, batch, batch_size):

		self.latest_network.get_model().train()
		optimizer.zero_grad()

		loss = 0.0
		policy_loss = 0.0
		value_loss = 0.0

		for (state, (target_value, target_policy)) in batch:
			
			predicted_policy, predicted_value = self.latest_network.inference(state, True, self.config.num_training_iters)

			target_policy = torch.tensor(target_policy).to(self.latest_network.device)
			target_value = torch.tensor(target_value).to(self.latest_network.device)

			
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
	
	def batch_update_weights(self, optimizer, scheduler, batch, batch_size):

		self.latest_network.get_model().train()
		optimizer.zero_grad()

		loss = 0.0
		policy_loss = 0.0
		value_loss = 0.0

		states, targets = list(zip(*batch))
		values, policies = list(zip(*targets))

		batch_input = torch.cat(states, 0)

		predicted_policies, predicted_values = self.latest_network.inference(batch_input, True, self.config.num_training_iters)

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

	def train_network(self, optimizer, scheduler):
		print()
		
		# TODO: PARALLEL TRAINING
		batch_size = self.config.batch_size

		replay_size = ray.get(self.replay_buffer.len.remote(), timeout=30)
		n_games = ray.get(self.replay_buffer.played_games.remote(), timeout=30)
		if self.config.test_set:
			test_size = ray.get(self.test_buffer.len.remote(), timeout=30)

		print("\nAfter Self-Play there are a total of " + str(replay_size) + " positions in the replay buffer.")
		print("Total number of games: " + str(n_games))

		start = time.time()
		
		if self.config.learning_method == "epochs":
			
			
			if  batch_size > replay_size:
				print("Batch size too large.\n" + 
					"If you want to use batche_size with more moves than the first batch of games played " + 
					"you need to use, the \"early_fill\" config to fill the batch with random moves at the beggining.\n")
				exit()
			else:
				number_of_batches = replay_size // batch_size
				print("Batches in replay buffer: " + str(number_of_batches))

				if self.config.test_set:
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

			
			values = [2e-1, 2e-2, 2e-3, 2e-4]
			momentum = 0.9

			epochs = self.config.learning_epochs
			total_updates = epochs*number_of_batches
			print("\nTotal number of updates: " + str(total_updates) + "\n")
			
			bar = ChargingBar('Training ', max=epochs)
			for e in range(epochs):

				ray.get(self.replay_buffer.shuffle.remote(), timeout=30)
				if self.config.test_set:
					ray.get(self.test_buffer.shuffle.remote(), timeout=30)

				epoch_value_loss = 0.0
				epoch_policy_loss = 0.0
				epoch_combined_loss = 0.0

				if self.config.test_set:
					t_epoch_value_loss = 0.0
					t_epoch_policy_loss = 0.0
					t_epoch_combined_loss = 0.0

				spinner = PieSpinner('\t\t\t\t\t\t  Running epoch ')
				for b in range(number_of_batches):		
					start_index = b*batch_size
					next_index = (b+1)*batch_size

					# TODO: get entire buffer and slice locally
					batch = ray.get(self.replay_buffer.get_slice.remote(start_index,next_index), timeout=55) # the slice does not take the last element
				
					value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler, batch, batch_size)

					epoch_value_loss += value_loss
					epoch_policy_loss += policy_loss
					epoch_combined_loss += combined_loss
					spinner.next()

				epoch_value_loss /= number_of_batches
				epoch_policy_loss /= number_of_batches
				epoch_combined_loss /= number_of_batches	
				
				if self.config.test_set:
					for t in range(number_of_test_batches):
						start_index = t*batch_size
						next_index = (t+1)*batch_size
						batch = self.test_buffer.get_slice.remote(start_index,next_index)
					
						test_value_loss, test_policy_loss, test_combined_loss = self.test_set_loss(batch, batch_size)

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

					if self.config.test_set:
						self.tests_value_loss.append(test_value_loss)
						self.tests_policy_loss.append(test_policy_loss)
						self.tests_combined_loss.append(test_combined_loss)

				if ((e+1) % 100 == 0):
					# sanity check, just to make sure policy isn't nan (np.random.choice crashes if any of the values is nan)
					predicted_policy, _ = self.latest_network.inference(self.game.generate_state_image(), False, self.config.num_pred_iters)
					probs = predicted_policy.cpu()[0].numpy().flatten()
					chance_action = np.random.choice(self.game.num_actions, p=probs)

				spinner.finish
				bar.next()
				
			bar.finish

			self.train_global_value_loss.extend(self.epochs_value_loss)
			self.train_global_policy_loss.extend(self.epochs_policy_loss)
			self.train_global_combined_loss.extend(self.epochs_combined_loss)

			if self.config.test_set:
				self.test_global_value_loss.extend(self.tests_value_loss)
				self.test_global_policy_loss.extend(self.tests_policy_loss)
				self.test_global_combined_loss.extend(self.tests_combined_loss)
				
		else:
			future_buffer = self.replay_buffer.get_buffer.remote()
			batch=[]
			probs=[]
			if self.config.late_heavy:
				# The way I found to create a scalling array
				num_positions = replay_size
				variation = 0.6
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
			print("\nTotal number of updates: " + str(self.config.num_samples) + "\n")

			buffer = ray.get(future_buffer, timeout=30)
			for _ in range(self.config.num_samples):
				if probs == []:
					batch_indexes = np.random.choice(len(buffer), size=batch_size, replace=True)
				else:
					batch_indexes = np.random.choice(len(buffer), size=batch_size, replace=True, p=probs)
				
				batch = [buffer[i] for i in batch_indexes]

				value_loss, policy_loss, combined_loss = self.batch_update_weights(optimizer, scheduler, batch, batch_size)

				average_value_loss += value_loss
				average_policy_loss += policy_loss
				average_combined_loss += combined_loss

			self.train_global_value_loss.append(average_value_loss/self.config.num_samples)
			self.train_global_policy_loss.append(average_policy_loss/self.config.num_samples)
			self.train_global_combined_loss.append(average_combined_loss/self.config.num_samples)



		
		end = time.time()
		total_time = end-start
		print("\n\nTraining time(s): " + format(total_time, '.4') + "\n")
		return
			
	def run_selfplay(self, num_games_per_batch, test_set, show, text="Self-Play"):
		start = time.time()
		print("\n")

		if test_set:
			buffer_to_use = self.test_buffer
		else:
			buffer_to_use = self.replay_buffer

		chunk_size = 128
		num_chunks = num_games_per_batch // chunk_size
		rest = num_games_per_batch % chunk_size

		bar = ChargingBar(text, max=num_games_per_batch)
		for c in range(num_chunks+1):
			games_to_play = chunk_size
			if c == num_chunks:
				games_to_play = rest

			actor_list = [Gamer.remote(buffer_to_use, self.shared_storage, self.config, self.game_class, self.game_args) for a in range(self.config.num_actors)]
			actor_pool = ray.util.ActorPool(actor_list)

			for g in range(games_to_play):
				#show = True if (g+1)% 1111 == 0 else False
				actor_pool.submit(lambda actor, args: actor.play_game.remote(args), show)

		
			for g in range(games_to_play):
				actor_pool.get_next_unordered() # Set Timeout and Ignore_if_timeout when running continuous training
				bar.next()
	
		bar.finish
		

		end = time.time()
		total_time = end-start
		print("\n\nTotal time(s): " + format(total_time, '.6'))
		print("Average time per game(s): " + format(total_time/num_games_per_batch, '.4'))

		return
	
	def run_tests(self, AI_player, num_games, show_results=True, text="Testing"):	
		start = time.time()
		print("\n")
		
		wins = [0,0]

		chunk_size = 128
		num_chunks = num_games // chunk_size
		rest = num_games % chunk_size

		args_list = [AI_player, None, self.latest_network]

		
		bar = ChargingBar(text, max=num_games)
		for c in range(num_chunks+1):
			games_to_play = chunk_size
			if c == num_chunks:
				games_to_play = rest

			actor_list = [RemoteTester.remote
		 		(
				self.config.num_testing_iters, 
				self.config.mcts_simulations, 
				self.config.pb_c_base, 
				self.config.pb_c_init, 
				False
				) 
				for a in range(self.config.num_actors)]
			

			actor_pool = ray.util.ActorPool(actor_list)

			for g in range(games_to_play):
				game = self.game_class(*self.game_args)
				args_list[1] = game
				actor_pool.submit(lambda actor, args: actor.Test_AI_with_mcts.remote(*args), args_list)

		
			for g in range(games_to_play):
				winner = actor_pool.get_next_unordered() # Set Timeout and Ignore_if_timeout when running continuous training
				if winner != 0:
					wins[winner-1] +=1
				bar.next()
			
	
		bar.finish

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


		#bad_moves_ratio = invalid/decision_count
		#decisions_per_game = decision_count / NUMBER_OF_GAMES
		#invalids_per_game = invalid / NUMBER_OF_GAMES

		if show_results:
			print("\n\nAI playing as p" + str(AI_player) + "\n")

			print("P1 Win ratio: " + format(p1_winrate, '.4'))
			print("P2 Win ratio: " + format(p2_winrate, '.4'))
			print("Draw percentage: " + format(draw_percentage, '.4'))
			print("Comparative Win ratio(p2/p1): " + cmp_2_string)
			print("Comparative Win ratio(p1/p2): " + cmp_1_string + "\n", flush=True)


		end = time.time()
		total_time = end-start
		print("\n\nTotal testing time(s): " + format(total_time, '.6'))
		print("Average time per game(s): " + format(total_time/num_games, '.4'))

		return p1_winrate, p2_winrate, draw_percentage



	








