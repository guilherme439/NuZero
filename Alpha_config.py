


class Alpha_Zero_config():

	def __init__(self):

		# Optimization
		self.parallel_testing = True
		self.with_cache = True
		self.num_actors = 6


		# Run Settings
		self.early_fill = 1000
		self.num_games_per_batch = 200
		self.num_batches = 40

		self.save_frequency = 1
		self.test_frequency = 1
		self.debug_frequency = 1
		self.storage_frequency = 1
		self.plot_frequency = 1
		self.plot_reset = 250
		self.plot = False

		# Loss Test-set
		self.test_set = False
		self.num_test_set_games = 50

		# Play Testing
		self.num_wr_testing_games = 100

		# Recurrent options
		self.num_training_iters = 2
		self.num_pred_iters = 2
		self.num_testing_iters = 2
		
		# Learning
		self.shared_storage_size = 5

		self.replay_window_size = int(2000)
		self.batch_size = 16

		self.learning_method = "epochs" 	# epochs | samples

		## Epochs learning method
		self.learning_epochs = 16

		## Samples learning method
		self.num_samples = 1
		self.late_heavy = True

		# Optimizer
		self.optimizer = "Adam"
		
		self.learning_rate = 2e-5
		self.scheduler_boundaries = [100e3, 600e3, 1500e3]
		self.scheduler_gama = 0.1

		## More Optimizer
		self.weight_decay = 0.0
		self.momentum = 0.9


		# Search
		# Not recomended to use a number os mcts simulations N such that sqrt(N) <= N / average_possible_actions. Because... decision criteria during MCTS. 
		self.mcts_simulations = 800
		self.keep_sub_tree = True # Recommended to set as False for simpler games in order to make tunning and optimization easier
		self.use_terminal = False # Wheter to use the actual game's results at terminal nodes during the mcts for stronger "signals" during early policy training.
		
		
		self.pb_c_base = 19652 # this value should be based on the number of mcts_simulations and average game lenght
		self.pb_c_init = 1.25 # relative importance to give to the prior probability
		# Since less visited nodes will have a lower bias discount ths parameter also controls the level of exploration


		# The pb_c bias value added by the deepmind team is used to counter act the increased importance of the value in the later stages of the game

		# Important to tune the prior score as will as the pb_c bias value to determine when to give more importance to value/prior


		# Exploration
		self.number_of_softmax_moves = 5
		self.epsilon_random_exploration = 0.0001


		# Means
		#
		# Gamma distribution: alpha * beta
		# Beta distribution: alpha / (alpha + beta) (the bigger beta is the more "agressive" the distribution is)
		# Normal distribution: alpha

		# ---------------------------------------------------------

		# Beta dist good values
		#
		# Mean: 0.1 -> alpha = x beta = x
		# Mean: 0.2 -> alpha = 

		self.root_exploration_destribution = "gamma" # gamma | beta | normal
		self.root_exploration_fraction = 0.2
		self.dist_alpha = 0.2
		self.dist_beta = 1 # The original alpha_zero always used beta=1 and varied alpha acording to the number of legal moves
		
	def set_tic_tac_toe_config(self):
		
		# Optimization
		self.parallel_testing = True
		self.with_cache = True
		self.num_actors = 6


		# Run Settings
		self.early_fill = 1200
		self.num_games_per_batch = 200
		self.num_batches = 200

		self.save_frequency = 1
		self.test_frequency = 1
		self.debug_frequency = 1
		self.storage_frequency = 1
		self.plot_frequency = 1
		self.plot_reset =50
		self.plot = False

		# Loss Test-set
		self.test_set = False
		self.num_test_set_games = 50

		# Play Testing
		self.num_wr_testing_games = 100

		# Recurrent options
		self.num_training_iters = 2
		self.num_pred_iters = 2
		self.num_testing_iters = 2
		
		# Learning
		self.shared_storage_size = 2

		self.replay_window_size = int(3000)
		self.batch_size = 16

		self.learning_method = "epochs" 	# epochs | samples

		## Epochs learning method
		self.learning_epochs = 16

		## Samples learning method
		self.num_samples = 256
		self.late_heavy = True

		# Optimizer
		self.optimizer = "Adam"
		
		self.learning_rate = 2e-5
		self.scheduler_boundaries = [100e3, 600e3, 1500e3]
		self.scheduler_gama = 0.1

		## More Optimizer
		self.weight_decay = 0.0
		self.momentum = 0.9

		# Search
		self.mcts_simulations = 50
		self.keep_sub_tree = False
		self.use_terminal = True
		
		self.pb_c_base = 500
		self.pb_c_init = 1.0

		# Exploration
		self.number_of_softmax_moves = 6
		self.epsilon_random_exploration = 0.0001


		self.root_exploration_distribution = "gamma"
		self.root_exploration_fraction = 0.0
		self.dist_alpha = 0.0
		self.dist_beta = 1


		return
	
	def set_SCS_config(self):

		# Optimization
		self.parallel_testing = True
		self.with_cache = True
		self.num_actors = 5


		# Run Settings
		self.early_fill = 50
		self.num_games_per_batch = 100
		self.num_batches = 50

		self.save_frequency = 1
		self.test_frequency = 1
		self.debug_frequency = 999
		self.storage_frequency = 1
		self.plot_frequency = 1
		self.plot_reset = 200
		self.plot = False

		# Loss Test-set
		self.test_set = False
		self.num_test_set_games = 50

		# Play Testing
		self.num_wr_testing_games = 40

		# Recurrent options
		self.num_training_iters = 3
		self.num_pred_iters = 3
		self.num_testing_iters = 3
		
		# Learning
		self.shared_storage_size = 5

		self.replay_window_size = int(800)
		self.batch_size = 16

		self.learning_method = "epochs" 	# epochs | samples

		## Epochs learning method
		self.learning_epochs = 8

		## Samples learning method
		self.num_samples = 512
		self.late_heavy = True

		# Optimizer
		self.optimizer = "Adam"
		
		self.learning_rate = 2e-4
		self.scheduler_boundaries = [100e3, 600e3, 1500e3]
		self.scheduler_gama = 0.1

		## More Optimizer
		self.weight_decay = 0.0
		self.momentum = 0.9

		# Search
		self.mcts_simulations = 250
		self.keep_sub_tree = True
		self.use_terminal = True
		
		self.pb_c_base = 12000
		self.pb_c_init = 0.95

		## Exploration
		self.number_of_softmax_moves = 40
		self.epsilon_random_exploration = 0.0


		self.root_exploration_distribution = "gamma"
		self.root_exploration_fraction = 0.1
		self.dist_alpha = 0.1
		self.dist_beta = 1



		return
	
	def save_config(self, path, network_name):

		f = open(path, "w+")

		f.write("\n")
		f.write("\n|Results for " + network_name + "\n|")

		f.write("\nNumber of batches: " + str(self.num_batches))
		f.write("\nGames per batch: " + str(self.num_games_per_batch))
		f.write("\nreplay_window_size: " + str(self.replay_window_size))
		f.write("\nbatch_size: " + str(self.batch_size))
		f.write("\nlearning_method: " + str(self.learning_method))
		f.write("\noptimizer: " + str(self.optimizer))
		f.write("\nlearning_epochs: " + str(self.learning_epochs))
		f.write("\nlearning_rate: " + str(self.learning_rate))
		f.write("\nweight_decay: " + str(self.weight_decay))
		f.write("\nmcts_simulations: " + str(self.mcts_simulations))
		f.write("\nkeep_sub_tree: " + str(self.keep_sub_tree))
		f.write("\npb_c_base: " + str(self.pb_c_base))
		f.write("\npb_c_init: " + str(self.pb_c_init))
		f.write("\nnumber_of_softmax_moves: " + str(self.number_of_softmax_moves))
		f.write("\nepsilon_random_exploration: " + str(self.epsilon_random_exploration))
		f.write("\nroot_exploration_distribution: " + str(self.root_exploration_distribution))
		f.write("\nroot_exploration_fraction: " + str(self.root_exploration_fraction))
		f.write("\ndist_alpha: " + str(self.dist_alpha))
		f.write("\ndist_beta: " + str(self.dist_beta))
		f.write("\n")

		f.close()

		return



	

	
	
