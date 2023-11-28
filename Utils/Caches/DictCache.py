
from Utils.Caches.Cache import Cache

class DictCache(Cache):
    ''' This cache is just a wrapper for a python dictionary '''

    def __init__(self):
        self.dict = {}
        return
    
    def contains(self, tensor_key):
        key = tuple(tensor_key.numpy().flatten())
        value = self.dict.get(key)
        return value is not None
    
    def get(self, tensor_key):
        '''Returns the value for the key, or None if the key doesn't exist'''
        key = tuple(tensor_key.numpy().flatten())
        return self.dict.get(key)
    
    def put(self, item):
        (tensor_key, value) = item
        key = tuple(tensor_key.numpy().flatten())
        self.dict[key] = value
        return
    
    def length(self):
        ''' Returns the number of items in the cache '''
        return len(self.dict)


    
    

    
