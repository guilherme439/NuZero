from .Config import Config

class Training_Config(Config):

    def __init__(self):

        self.running = dict \
        (
        state_cache = "disabled",		# disabled | per_game | per_actor
        # Wether to keep a cache within each actor/game of previously seen states, to avoid using slow network inference.
        # Consumes a lot of memory so it is only worth using if the game being played repeats states frequently.
        
        num_actors = 4,
        early_fill = 1000,
        early_softmax_exploration = 1,
        early_random_exploration = 0,
    	# Number of games to be played until actors are replaced with new ones.
        )


        self.sequential = dict \
        (
        num_games_per_batch = 200,
        num_batches = 40,
        )


        self.asynchronous = dict \
        (
        training_steps = 1e5,
        early_delay = 6000,
        update_delay = 1600,
        )
        

        self.saving = dict \
        (
        save_frequency = 1,
        storage_frequency = 1,
        )


        self.testing = dict \
        (
        early_testing = True,
        policy_test_frequency = 10,
        mcts_test_frequency = 80,
        num_policy_test_games = 100,
        num_mcts_test_games = 100,
        )


        self.plotting = dict \
        (
        plot_frequency = 50,
        plot_reset = 0,
        )


        self.recurrent_networks = dict \
        (
        num_train_iterations = 2,
        num_pred_iterations = 2,
        num_test_iterations = 2
        )
        
        
        self.learning = dict \
        (
        shared_storage_size = 5,
        replay_window_size = int(2000),
        batch_extraction = 'local',     # local | remote
        value_loss = "SE",              # SE | AE
        policy_loss = "CEL",            # CEL | KLD | MSE
        normalize_CEL = False,          # Cross entropy loss normalization
        skip_target = 'policy',         # policy | value
        skip_frequency = 0,             # How often loss should be calculated ignoring the "skip_target"
        learning_method = "epochs"      # epochs | samples
        )


        # Epochs learning method	
        self.epochs = dict \
        (
        batch_size = 32,
        learning_epochs = 8,
        plot_epoch = False,
        test_set = False,
        num_test_set_games = 0
        )


        # Samples learning method
        self.samples = dict   \
        (
        batch_size = 2024,
        num_samples = 1,
        late_heavy = False
        )


        self.optimizer = dict \
        (
        optimizer = "SGD",      # SGD | Adam
        learning_rate = 2e-1,

        # SGD Optimizer
        weight_decay = 1e-5,
        momentum = 0.9,

        # Scheduler
        scheduler_boundaries = [100e3, 600e3, 1500e3],
        scheduler_gamma = 0.1
        )

        return