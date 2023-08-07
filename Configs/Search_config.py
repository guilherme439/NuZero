from .Config import Config

class Search_config(Config):


    def __init__(self):

            
        self.simulation = dict \
        (
        mcts_simulations = 800,	# Not recomended to use a number os mcts simulations N such that sqrt(N) <= N/average_possible_actions. Because... decision criteria during MCTS.
        keep_sub_tree = True,	# Recommended to set as False for simpler games in order to make tunning and optimization easier
        )
			
		
        self.uct = dict \
        (
        pb_c_base = 19652, # this value should be based on the number of mcts_simulations and average game lenght
        pb_c_init = 1.25 # relative importance to give to the prior probability
        # Since less visited nodes will have a lower bias discount this parameter also controls the level of exploration
        )

        # pb_c_base and pb_c_init will be use to calculate the pb_c bias.
        # You can check the formula in the function "score()" or in the paper.

        # The pb_c bias value added by the deepmind team is used to
        # counter act the increased importance of the value in the later stages of the game
        # and to increase exploration as the game progresses


        self.exploration = dict \
        (
        number_of_softmax_moves = 5,
        epsilon_random_exploration = 0.0001,
        value_factor = 1.0,
        root_exploration_destribution = "gamma", # gamma | beta | normal
        root_exploration_fraction = 0.2,
        dist_alpha = 0.2,
        dist_beta = 1 # The original alpha_zero always used beta=1 and varied alpha acording to the number of legal moves

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
		)


		# Important to tune the noise in the prior score as well as the pb_c bias factor,
        # to determine the importance given to the value/prior given by the network throughout the search
		

			
	
