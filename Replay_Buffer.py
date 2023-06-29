import ray
import random
import math
import numpy as np

from Progress_bar import Progress_bar


@ray.remote(scheduling_strategy="SPREAD")
class Replay_Buffer():

    def __init__(self, window_size, batch_size):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer = []
        self.n_games = 0


    def save_game(self, game):
        full = False
        if self.n_games >= self.window_size:
            full = True
        else:
            self.n_games += 1

        for i in range(len(game.state_history)):
            state = game.get_state_from_history(i)
            pair = (state, game.make_target(i))
            if full:
                self.buffer.pop(0)
            self.buffer.append(pair)


    def shuffle(self):
        random.shuffle(self.buffer)

    def get_slice(self, start_index, last_index):
        return self.buffer[start_index:last_index]
    
    def get_buffer(self):
        return self.buffer

    def len(self):
        return len(self.buffer)
    
    def played_games(self):
        return self.n_games

    

    
