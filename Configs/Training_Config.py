from .Config import Config

class Training_Config(Config):

    def __init__(self):

        self.running = dict \
        (
        cache_choice = "disabled",		# disabled | dict | keyless       
        running_mode = "sequential",
        num_actors = 4,
        early_fill_per_type = 250,
        early_softmax_exploration = 1,
        early_random_exploration = 0,
    	training_steps = 1e5,
        )

        self.sequential = dict \
        (
        num_games_per_step = 200,
        )


        self.asynchronous = dict \
        (
        update_delay = 1600,
        )


        self.cache = dict \
        (
        cache_choice = 'disabled',
        size_estimate = 10000
        )
        

        self.saving = dict \
        (
        save_frequency = 1,
        storage_frequency = 1,
        )


        self.testing = dict \
        (
        asynchronous_testing = True,
        testing_actors = 2,
        early_testing = True,
        policy_test_frequency = 10,
        mcts_test_frequency = 80,
        num_policy_test_games = 100,
        num_mcts_test_games = 100,
        test_iterations = 4,
        test_game_index = 1
        )


        self.plotting = dict \
        (
        plot_loss = True,  
        plot_weights = False,  
        plot_frequency = 50,
        value_split = 0,
        policy_split = 0,
        combined_split = 0,
        )


        self.recurrent_training = dict \
        (
        train_iterations = [4],
        pred_iterations = [4],
        alpha = 0.5
        )
        
        
        self.learning = dict \
        (
        shared_storage_size = 5,
        replay_window_size = int(2000),
        batch_extraction = 'local',     # local | remote
        value_loss = "SE",              # SE | AE
        policy_loss = "CEL",            # CEL | KLD | MSE
        normalize_CEL = False,          # Cross entropy loss normalization
        learning_method = "epochs"      # epochs | samples
        )


        # Epochs learning method	
        self.epochs = dict \
        (
        batch_size = 32,
        learning_epochs = 8,
        plot_epochs = False,
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
        nesterov = 0.1,
        # If you use SGD optimizer, it applies L2 weight regularization

        # Lr Scheduler
        scheduler_boundaries = [100e3, 600e3, 1500e3],
        scheduler_gamma = 0.1
        )

        return