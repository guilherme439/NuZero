from .Config import Config

class AlphaZero_config(Config):

    def __init__(self):


        self.optimization = dict \
        (
        state_cache = "per_game"		# disabled | per_game | per_actor
        # Wether to keep a cache within each actor/game of previously seen states, to avoid using slow network inference.
        # Consumes a lot of memory so it is only worth using if the game being played repeats states frequently.
        )


        self.actors = dict \
        (
        num_actors = 4,
        chunk_size = 128		# Number of games to be played until actors are replaced with new ones.
        )


        self.running = dict \
        (
        early_fill = 1000,
        num_games_per_batch = 200,
        num_batches = 40,

        testing_mode = "policy",        # policy | mcts
        num_wr_testing_games = 100,

        test_set = False,
        num_test_set_games = 50
        )


        self.frequency = dict \
        (
        save_frequency = 1,
        test_frequency = 1,
        debug_frequency = 1,
        storage_frequency = 1,
        plot_frequency = 1,
        plot_reset = 250,
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
        learning_method = "epochs",     # epochs | samples
        normalize_loss = False          # Policy loss normalization
        )


        # Epochs learning method	
        self.epochs = dict \
        (
        batch_size = 32,
        learning_epochs = 8,
        plot_epoch = False
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