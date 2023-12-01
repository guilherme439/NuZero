import ray
import random
import math
import numpy as np


@ray.remote(scheduling_strategy="SPREAD")
class ReplayBuffer():

    def __init__(self, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer = []
        self.n_games = 0


    def save_game(self, game, game_index):
        full = False
        if self.n_games >= self.window_size:
            full = True
        else:
            self.n_games += 1

        for i in range(len(game.state_history)):
            state = game.get_state_from_history(i)
            tuple = (state, game.make_target(i), game_index)
            if full:
                self.buffer.pop(0)
            self.buffer.append(tuple)


    def shuffle(self):
        random.shuffle(self.buffer)

    def get_slice(self, start_index, last_index):
        return self.buffer[start_index:last_index]
    
    def get_sample(self, batch_size, replace, probs):
        if probs == []:
            args = [len(self.buffer), batch_size, replace]
        else:
            args = [len(self.buffer), batch_size, replace, probs]
        
        batch_indexes = np.random.choice(*args)
        batch = [self.buffer[i] for i in batch_indexes]

        return batch
    
    def get_buffer(self):
        return self.buffer

    def len(self):
        return len(self.buffer)
    
    def played_games(self):
        return self.n_games

    

    
