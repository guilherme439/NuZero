import ray


@ray.remote(scheduling_strategy="SPREAD")
class Shared_network_storage():

    def __init__(self, window_size):
        self.network_list = []
        self.window_size = window_size

    def get_latest_network(self):
        return self.network_list[-1]
    
    def save_network(self, network):
        if len(self.network_list) >= self.window_size:
            self.network_list.pop(0)
        
        self.network_list.append(network)
