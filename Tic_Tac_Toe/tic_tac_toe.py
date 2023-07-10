import copy
import numpy as np
import torch
import enum
import sys
import gc


from collections import Counter
from termcolor import colored





class tic_tac_toe():
    WIDTH = 3
    HEIGHT = 3
    TURNS = 9

    N_PLAYERS = 2


    def __init__(self):
        self.board = np.zeros((3,3))
        self.current_player = 1  # Two players: 1 and 2
        
        self.total_action_planes = 1
        self.action_space_shape = (self.total_action_planes , self.HEIGHT , self.WIDTH)
        self.num_actions = 9
        
        self.game_state_shape = (2,3,3)
       
        # MCTS support atributes
        self.terminal = False
        self.length = 0
        self.terminal_value = 0
        self.child_policy = []
        self.state_history = []
        self.player_history = []
        self.action_history = []

        self.reset_env()
        
        return

##########################################################################
# -------------------------                    ------------------------- #
# -----------------------  GET AND SET METHODS  ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def getBoardWidth(self):
        return self.WIDTH

    def getBoardHeight(self):
        return self.HEIGHT
    
    def get_length(self):
        return self.length
    
    def get_current_player(self):
        return self.current_player

    def get_action_space_shape(self):
        return self.action_space_shape  
    
    def get_num_actions(self):
        return self.num_actions

    def state_shape(self):
        return self.game_state_shape

    def get_terminal_value(self):
        return self.terminal_value
    
    def get_state_from_history(self, i):
        return self.state_history[i]
    
    def is_terminal(self):
        return self.terminal

    def store_state(self, state):
        self.state_history.append(state)
        return

    def store_player(self, player):
        self.player_history.append(player)
        return

    def store_action(self, action_coords):
        self.action_history.append(action_coords)

    def get_name(self):
        return "Tic_Tac_Toe"

##########################################################################
# ----------------------------              ---------------------------- #
# --------------------------  CORE FUNCTIONS  -------------------------- #
# ----------------------------              ---------------------------- #
##########################################################################

    def reset_env(self):
        self.board = np.zeros((3,3))
        self.current_player = 1  # Two players: 1 and 2

        # MCTS support atributes
        self.terminal = False
        self.length = 0
        self.terminal_value = 0
        self.child_policy = []
        self.state_history = []
        self.player_history = []
        self.action_history = []
        return
    
    def possible_actions(self):
        mask = np.ones((self.HEIGHT,self.WIDTH))

        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if self.board[i][j] != 0:
                    mask[i][j] = 0

        return mask

    def play_action(self, action_coords):
        self.board[action_coords[1]][action_coords[2]] = self.current_player
        return

    def generate_state_image(self):
        p1_pieces = np.zeros((3,3))
        p2_pieces = np.zeros((3,3))

        #'''
        player_plane = np.ones((3,3))
        if self.current_player == 2:
            player_plane.fill(-1)
        #'''
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if self.board[i][j] == 1:
                    p1_pieces[i][j] = 1
                elif self.board[i][j] == 2:
                    p2_pieces[i][j] = 1

        player_plane = torch.unsqueeze(torch.as_tensor(player_plane, dtype=torch.float32), 0)
        p1_pieces = torch.unsqueeze(torch.as_tensor(p1_pieces, dtype=torch.float32), 0)
        p2_pieces = torch.unsqueeze(torch.as_tensor(p2_pieces, dtype=torch.float32), 0)

        #stack_list = (player_plane, p1_pieces, p2_pieces)
        stack_list = (p1_pieces, p2_pieces)
        new_state = torch.concat(stack_list, dim=0)
        state_image = torch.unsqueeze(new_state, 0) # add batch size to the dimensions
        return state_image

    def step_function(self, action_coords):
        self.play_action(action_coords)
        self.length += 1
        _, done = self.check_victory()
        self.current_player = (self.length%2) + 1

        return done

    def check_victory(self):
        size = self.HEIGHT

        rewards = [0, 0]
        done = False
        final_value = 0

        p1_left_diagonal_count = 0
        p2_left_diagonal_count = 0
        p1_right_diagonal_count = 0
        p2_right_diagonal_count = 0

        p1_horizontal_counts = np.zeros(3)
        p1_vertical_counts = np.zeros(3)
        p2_horizontal_counts = np.zeros(3)
        p2_vertical_counts = np.zeros(3)

        for i in range(size):
            x = i
            y = (size-1) - x
            if self.board[x][y] == 1:
                p1_right_diagonal_count += 1
            elif self.board[x][y] == 2:
                p2_right_diagonal_count += 1
            
            if self.board[x][x] == 1:
                p1_left_diagonal_count += 1
            elif self.board[x][x] == 2:
                p2_left_diagonal_count += 1


            
            for j in range(size):
                if self.board[i][j] == 1:
                    p1_horizontal_counts[i] += 1
                elif self.board[i][j] == 2:
                    p2_horizontal_counts[i] += 1
                
                if self.board[j][i] == 1:
                    p1_vertical_counts[i] += 1
                elif self.board[j][i] == 2:
                    p2_vertical_counts[i] += 1

            
            if p1_left_diagonal_count == 3 or p1_right_diagonal_count == 3 or any(c == 3 for c in p1_horizontal_counts) or any(c == 3 for c in p1_vertical_counts):
                rewards = [1, -1]
                final_value = 1
                done = True
            elif p2_left_diagonal_count == 3 or p2_right_diagonal_count == 3 or any(c == 3 for c in p2_horizontal_counts) or any(c == 3 for c in p2_vertical_counts):
                rewards = [-1, 1]
                final_value = -1
                done = True

            if self.length == 9:
                done = True  

            if done:
                self.terminal_value = final_value
                self.terminal = True

            
        #print(p1_left_diagonal_count, p2_left_diagonal_count, p1_right_diagonal_count, p2_right_diagonal_count,
        #      p1_horizontal_counts, p1_vertical_counts, p2_horizontal_counts, p2_vertical_counts)

        return rewards, done

    def check_winner(self):
        #funny and overcomplicated way of calculating the winner of the game
        return ((1 - self.terminal_value) + (self.terminal_value * (self.terminal_value > 0))) * self.terminal_value * self.terminal_value

    def get_rewards_and_game_env(self):
        # not needed in this game (check victory does everything) maybe change it later
        return

    def store_search_statistics(self, node):
        sum_visits = sum(child.visit_count for child in node.children.values())
        self.child_policy.append([
            node.children[a].visit_count / sum_visits if a in node.children else 0
            for a in range(self.num_actions)
        ])

    def make_target(self, i):

        value_target = self.terminal_value
        policy_target = self.child_policy[i]

        target = (value_target, policy_target)
        return target

##########################################################################
# -------------------------                   -------------------------- #
# ------------------------  UTILITY FUNCTIONS  ------------------------- #
# -------------------------                   -------------------------- #
##########################################################################

    def clone(self):
        return copy.deepcopy(self)
    
    def shallow_clone(self):
        game_clone = tic_tac_toe()
        game_clone.board = copy.deepcopy(self.board)
        game_clone.current_player = copy.deepcopy(self.current_player)
        game_clone.length = copy.deepcopy(self.length)

        return game_clone


    def string_representation(self):
        string = ""
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                if self.board[i][j] == 1:
                    mark=colored(" O ", "blue")
                elif self.board[i][j] == 2:
                    mark=colored(" X ", "red")
                else:
                    mark = "   "

                if j != self.WIDTH-1:
                    string += mark  + '|'
                else:
                    string += mark
            
            string += "\n"
            if(i<self.HEIGHT-1):
                for k in range(self.WIDTH-1):
                    string += "---|"
                string += "---\n"
            
        return string

##########################################################################
# ------------------------                     ------------------------- #
# -----------------------  USER PLAY FUNCTIONS  ------------------------ #
# ------------------------                     ------------------------- #
##########################################################################

    def play_user_vs_user(self):

        while not self.terminal:
            self.print_board()
            x = input("choose coordenates: ")
            action_coords = eval(x)
            self.step_function(action_coords)

        self.print_board()

