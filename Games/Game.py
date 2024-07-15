import numpy as np

class Game():

    # Generic Game class
    def __init__(self):
        self.action_space_shape = (0,0,0)
        self.game_state_shape = (0,0,0)
        pass
    
    def get_state_shape(self):
        return self.game_state_shape

    def get_action_space_shape(self):
        return self.action_space_shape
    
    def get_name(self):
        print("get_name not implemented")
        pass

    def get_num_actions(self):
        print("get_num_actions not implemented")
        pass

    def get_current_player(self):
        print("get_current_player not implemented")
        pass

    def is_terminal(self):
        print("is_terminal not implemented")
        pass

    def get_state_from_history(self, i):
        print("get_state_from_history not implemented")
        pass

    def get_terminal_value(self):
        # return terminal value. For example: 1 if player_1 wins and -1 if player_2 wins
        print("get_terminal_value not implemented")
        pass

    def get_winner(self):
        # return 0, 1 or 2, if the game ended in a draw, p1 or p2 victory, respectively
        print("get_winner not implemented")
        pass

    def get_lenght(self):
        # game lenght (usually is the number of moves)
        print("get_lenght not implemented")
        pass

    def store_player(self):
        print("store_player not implemented")
        pass

    def store_state(self):
        print("store_state not implemented")
        pass

    def possible_actions(self):
        # return a one-hot encoded mask of the possible actions
        print("possible_actions not implemented")
        pass

    def step_function(self, action_coordinates):
        # action_coordinates are the location of the action in the action space
        print("step_function not implemented")
        pass

    def store_search_statistics(self, node):
        # stores the nodes visit counts to later be used in training
        print("store_search_statistics not implemented")
        pass

    def clone(self):
        # creates a clone of the entire game
        # watch out for mem leaks
        print("clone not implemented")
        pass

    def shallow_clone(self):
        # creates a simpler clone of the game we just the necessary information for simulation
        print("shallow_clone not implemented")
        pass

    def generate_state_image(self):
        # generate the current game state image to be passed to the neural net
        print("generate_state_image not implemented")
        pass

    def make_target(self):
        # return a target of the form (target_value, target_policy)
        print("make_target not implemented")
        pass

    def get_action_coords(self, action_i):
        action_coords = np.unravel_index(action_i, self.get_action_space_shape())
        return action_coords
    
    def get_action_index(self, action_coords):
        action_i = np.ravel_multi_index(action_coords, self.get_action_space_shape())
        return action_i
    
    def string_representation(self):
        # necessary only for visualization purposes
        pass
