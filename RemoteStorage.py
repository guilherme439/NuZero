import ray

@ray.remote
class RemoteStorage():

    def __init__(self, precious=None):
        self.precisous = precious # store my precious

    def get_item(self):
        return self.precisous
    
    def set_item(self, item):
        self.precisous = item
