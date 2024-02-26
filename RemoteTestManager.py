import ray

from TestManager import TestManager

@ray.remote
class RemoteTestManager(TestManager):
    '''Remote wrapper of TestManager'''
	
    def __init__(self, game_class, game_args, num_testers, shared_storage, keep_updated):
        super().__init__(game_class, game_args, num_testers, shared_storage, keep_updated)

    
    

    