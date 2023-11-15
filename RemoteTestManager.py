import ray

from TestManager import TestManager

@ray.remote
class RemoteTestManager(TestManager):
    '''Remote version of TestManager'''
	
    def __init__(self, game_class, game_args, train_config, search_config, shared_storage, state_set=None):
        super().__init__(game_class, game_args, train_config, search_config, shared_storage, state_set=state_set)

    
    

    