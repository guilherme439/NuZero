import io
import torch
import hashlib
import metrohash
import math

from bitstring import BitArray

from Utils.Caches.Cache import Cache

'''
The way it works:
For each item, it calculates an hash and splits it in two: Part_1 and Part_2
Part_1 is used to index the item in the hash table.
Part_2 is stored alongside the item in the table and is used to resolve conflicts.
'''



class KeylessCache(Cache):
    ''' Implements a cache without storing the keys.'''
    

    def __init__(self, size_estimate):
        if size_estimate <= 0:
            print("\nThe cache size must be larger than 0")
            exit()

        self.indexing_bits = int(math.floor(math.log2(size_estimate)) + 1)
        self.size = int(math.pow(2, self.indexing_bits))
        self.max_index = self.size - 1
        
        self.table = [[]] * self.size
        self.num_items = 0
        self.fill_ratio = 0
        
        if self.indexing_bits < 16:
            self.hash_function = self.hash_metro64
        elif self.indexing_bits < 32:
            self.hash_function = self.hash_metro128
        elif self.indexing_bits < 256:
            self.hash_function = self.hash_sha256
            if self.indexing_bits > 64:
                print("\nWARNING: Using more than 64 bits out of 256, for indexing.\n")
        else:
            print("Cache size too large.")
            exit()
        return
    
    def length(self):
        return self.num_items
    
    def contains(self, key):
        full_hash, index, identifier = self.hash(key)
        value = self.find_by_id(identifier, self.table[index])
        return value is not None
    
    def get(self, key):
        '''Returns the value for the key, or None if the key doesn't exist'''
        full_hash, index, identifier = self.hash(key)
        value = self.find_by_id(identifier, self.table[index])
        return value
    
    def find_by_id(self, id, entry_list):
        for (value, identifier) in entry_list:
            if id == identifier:
                return value    
        return None
    
    def put(self, item):
        (key, value) = item
        self.num_items += 1
        self.fill_ratio = self.num_items / self.size

        '''
        if self.fill_ratio > 0.9:
            print("WARNING: Cache usage over 90%")
        elif self.fill_ratio > 0.8:
            print("WARNING: Cache usage over 80%")
        elif self.fill_ratio > 0.75:
            print("WARNING: Cache usage over 75%")
        '''

        full_hash, index, identifier = self.hash(key)

        cache_entry = (value, identifier)
        self.table[index].append(cache_entry)   
        return
    
    def update(self, update_cache):
        if update_cache.size != self.size:
            print("\nCannot update using caches of different sizes.")
            exit()
        for i in range(self.size):
            update_slot = update_cache.table[i]
            my_slot = self.table[i]
            for item in update_slot:
                (value, identifier) = item
                local_value = self.find_by_id(identifier, my_slot)
                if local_value is None:
                    my_slot.append(item)
                else:
                    if local_value != value:
                        print("\n\n\n")
                        print(local_value)
                        print("\n\n\n\n\n")
                        print(value)
                        print("\n\nSomething wrong while updating!\n")
                        exit()
        
        return
    
    def get_fill_ratio(self):
        return self.fill_ratio
    
    def hash(self, torch_tensor):
        byte_hash = self.hash_function(torch_tensor)
        bit_hash = BitArray(bytes=byte_hash)
        index = bit_hash[:self.indexing_bits]
        rest = bit_hash[self.indexing_bits:]
        return bit_hash.uint, index.uint, rest.uint

    def hash_metro64(self, torch_tensor):
        mh = metrohash.MetroHash64()
        mh.update(torch_tensor.numpy())
        value = mh.digest()
        return value
    
    def hash_metro128(self, torch_tensor):
        mh = metrohash.MetroHash128()
        mh.update(torch_tensor.numpy())
        value = mh.digest()
        return value
    
    def hash_sha256(self, torch_tensor):
        buff = io.BytesIO()                                                                                                                                            
        torch.save(torch_tensor, buff)
        tensor_as_bytes = buff.getvalue()
        sha = hashlib.sha256()
        sha.update(tensor_as_bytes)
        value = sha.digest()
        return value
        


    
    

    
