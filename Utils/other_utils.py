    
from Utils.Caches.DictCache import DictCache
from Utils.Caches.KeylessCache import KeylessCache

def create_cache(cache_choice, max_size):
    if cache_choice == "dict":
        cache = DictCache(max_size)
    elif cache_choice == "keyless":
        cache = KeylessCache(max_size)
    elif cache_choice == "disabled":
        cache = None
    else:
        print("\nbad cache_choice")
        exit()
    return cache