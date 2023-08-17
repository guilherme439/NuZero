import math
import numpy as np
import torch
import time
import yaml
import io

from copy import copy, deepcopy

from enum import Enum
from collections import Counter
from termcolor import colored


from .Unit import Unit
from .Tile import Tile
from .Terrain import Terrain

'''
From the hexagly source code, this is how the board is converted
from hexagonal to ortogonal representation:


 __    __                                 __ __ __ __
/11\__/31\__  . . .                      |11|21|31|41| . . .
\__/21\__/41\                            |__|__|__|__| 
/12\__/32\__/ . . .        _______|\     |12|22|32|42| . . .
\__/22\__/42\             |         \    |__|__|__|__| 
   \__/  \__/             |_______  /                           
 .  .  .  .  .                    |/       .  .  .  .  .
 .  .  .  .    .                           .  .  .  .    .
 .  .  .  .      .                         .  .  .  .      .


'''

'''
This is how rows and collumns are defined for SCS Games.
This definition might be different from the examples in the hexagdly repository,
but I believe it makes more sense this way.


#   This is a row:
#    __    __                    __ __ __ __
#   /11\__/13\__     ----->     |11|12|13|14|       - Rows are horizontal
#   \__/12\__/14\               |__|__|__|__|
#      \__/  \__/
#

#   And this is a column:
#    __                          __
#   /11\                        |11|
#   \__/        ----->          |__|                - Columns are vertical
#   /21\                        |21|
#   \__/                        |__|
#

'''

class SCS_Game():

    PHASES = 3              # Placement, Movement, Fighting
    STAGES = 6              # P1 Placement, P2 Placement, P1 Movement, P1 Fighting, P2 Movement, P2 Fighting

    N_PLAYERS = 2

    N_UNIT_STATUSES = 3     # Available, Moved, Attacked
    N_UNIT_STATS = 3        # Attack , Defense, Movement
    


    def __init__(self, game_config_path=""):
        
        # ------------------------------------------------------------ #
        # --------------------- INITIALIZATION  ---------------------- #
        # ------------------------------------------------------------ #
        self.turns = 0

        self.rows = 0
        self.columns = 0

        self.board = []
        self.current_player = 1
        self.current_phase = 0
        self.current_stage = 0   
        self.current_turn = 1
        
        self.available_units = [[],[]]
        self.moved_units = [[],[]]
        self.atacked_units = [[],[]]

        self.p1_last_index = 0
        self.p2_first_index = self.columns - 1

        self.victory_p1 = []
        self.victory_p2 = []
        self.n_vp = [0, 0]

        self.all_reinforcements = {0:[], 1:[]}
        self.current_reinforcements = {0:[], 1:[]}
        self.max_reinforcements = 0

        self.length = 0
        self.terminal_value = 0
        self.terminal = False
        
        if game_config_path != "":
            self.load_from_config(game_config_path)
            self.update_game_env()

        # ------------------------------------------------------ #
        # --------------- MCTS RELATED ATRIBUTES --------------- #
        # ------------------------------------------------------ #

        self.child_policy = []
        self.state_history = []
        self.player_history = []
        self.action_history = []

        # ------------------------------------------------------------ #
        # ------------- SPACE AND ACTION REPRESENTATION -------------- #
        # ------------------------------------------------------------ #

        ## ACTION REPRESENTATION

        # Number of planes per type
        self.placement_planes = 1
        self.movement_planes = 6 # The 6 sizes of the hexagon 
        self.fight_planes = 6 # The 6 sizes of the hexagon 
        self.no_move_planes = 1
        self.no_fight_planes = 1

        self.total_action_planes = \
        self.placement_planes + \
        self.movement_planes + \
        self.fight_planes + \
        self.no_move_planes + \
        self.no_fight_planes
        
        self.action_space_shape = (self.total_action_planes , self.rows , self.columns)
        self.num_actions     =     self.total_action_planes * self.rows * self.columns


        ## STATE REPRESENTATION
        self.n_reinforcement_turns = 3  # Number of turns for which the reinforcement are represented.
                                        # If this is 1, only the current turn is represented. 

        n_vic_dims = self.N_PLAYERS
        n_unit_dims = self.N_UNIT_STATS * self.N_UNIT_STATUSES * self.N_PLAYERS
        n_reinforcement_dims = self.n_reinforcement_turns * self.N_UNIT_STATS * self.N_PLAYERS
        n_feature_dims = 3 # turn, phase and player
        n_terrain_dims = 3 # atack, defense, movement
        total_dims = n_vic_dims + n_reinforcement_dims + n_unit_dims + n_feature_dims + n_terrain_dims

        self.game_state_shape = (total_dims, self.rows, self.columns)

        return
    
##########################################################################
# -------------------------                    ------------------------- #
# -----------------------  GET AND SET METHODS  ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def get_board(self):
        return self.board
    
    def getBoardColumns(self):
        return self.columns

    def getBoardRows(self):
        return self.rows    

    def get_action_space_shape(self):
        return self.action_space_shape

    def get_num_actions(self):
        return self.num_actions

    def get_current_player(self):
        return self.current_player
    
    def get_terminal_value(self):
        return self.terminal_value

    def is_terminal(self):
        return self.terminal

    def get_length(self):
        return self.length

    def get_state_from_history(self, i):
        return self.state_history[i]

    def state_shape(self):
        return self.game_state_shape

    def store_state(self, state):
        self.state_history.append(state)
        return

    def store_player(self, player):
        self.player_history.append(player)
        return
    
    def store_action(self, action_coords):
        self.action_history.append(action_coords)

    def get_name(self):
        return "SCS"
          
##########################################################################
# ----------------------------              ---------------------------- #
# ---------------------------   GAME LOGIC   --------------------------- #
# ----------------------------              ---------------------------- #
##########################################################################

    def step_function(self, action_coords):
        self.store_action(action_coords)
        self.play_action(action_coords)
        self.length += 1
        done = self.update_game_env()
        return done

    # ----------------- ACTIONS ----------------- #

    def possible_actions(self):
        player = self.current_player
        phase = self.current_phase
        size = self.rows * self.columns
        
        placement_plane = np.zeros((1, self.rows, self.columns), dtype=np.int32)
        movement_planes = np.zeros((self.movement_planes, self.rows, self.columns), dtype=np.int32)
        fight_planes = np.zeros((self.fight_planes, self.rows, self.columns), dtype=np.int32)
        no_move_plane = np.zeros((1, self.rows, self.columns), dtype=np.int32)
        no_fight_plane = np.zeros((1, self.rows, self.columns), dtype=np.int32)
        
        if (phase == 0):
            # Currently reinforcements can be placed on each player's side of the board
            available_columns = self.p1_last_index + 1 # number of columns on my side of the board
            my_half = np.ones((self.rows, available_columns), dtype=np.int32)
            rest_of_columns = self.columns - available_columns
            enemy_half = np.zeros((self.rows, rest_of_columns), dtype=np.int32)
                
            if player == 1:
                placement_plane = np.concatenate((my_half, enemy_half), axis=1)
            else:
                placement_plane = np.concatenate((enemy_half, my_half), axis=1)            

            for p in [0,1]:
                for unit in self.available_units[p]:
                    x = unit.row
                    y = unit.col
                    placement_plane[x][y] = 0
            # can not place on top of other units

            placement_plane = np.expand_dims(placement_plane, 0)
            
        
        if (phase == 1):            
            for unit in self.available_units[player-1]:
                x = unit.row
                y = unit.col

                no_move_plane[0][x][y] = 1 # no move action

                tiles = self.check_tiles((x,y))
                movements = self.check_mobility(unit, consider_other_units=True)

                for i in range(len(tiles)):
                    tile = tiles[i]
                    if (tile):
                        if(movements[i]):
                            movement_planes[i][x][y] = 1
                 
        if (phase == 2):
            for unit in self.moved_units[player-1]:
                row = unit.row
                col = unit.col
                
                no_fight_plane[0][row][col] = 1 # no fight action

                enemy_player = (not(player-1)) + 1 
                _ , enemy_dir = self.check_adjacent_units(unit, enemy_player)

                for direction in enemy_dir:
                    fight_planes[direction][row][col] = 1


        planes_list = [placement_plane, movement_planes, fight_planes, no_move_plane, no_fight_plane]
        valid_actions_mask = np.concatenate(planes_list)
        return valid_actions_mask
                      
    def parse_action(self, action_coords):
        act = None           # Represents the type of action
        start = (None, None) # Starting point of the action
        dest = (None, None)  # Destination point for the action

        current_plane = action_coords[0]

        # PLACEMENT PLANES
        if current_plane < self.placement_planes:
            act = 0
            start = (action_coords[1], action_coords[2])

        # MOVEMENT PLANES
        elif current_plane < (self.placement_planes + self.movement_planes):
            act = 1
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

            odd_col = y % 2
            plane_index = current_plane-self.placement_planes
            # n, ne, se, s, sw, nw
            if plane_index == 0:    # N
                dest = self.get_n_coords(start)

            elif plane_index == 1:  # NE
                dest = self.get_ne_coords(start)

            elif plane_index == 2:  # SE
                dest = self.get_se_coords(start)

            elif plane_index == 3:  # S
                dest = self.get_s_coords(start)

            elif plane_index == 4:  # SW
                dest = self.get_sw_coords(start)

            elif plane_index == 5:  # NW
                dest = self.get_nw_coords(start)
            else:
                print("Problem parsing action...Exiting")
                exit()
            
        # FIGHT PLANES
        elif current_plane < (self.placement_planes + self.movement_planes + self.fight_planes):
            act = 2
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

            odd_col = y % 2
            plane_index = current_plane - (self.placement_planes + self.movement_planes)
            # n, ne, se, s, sw, nw
            if plane_index == 0:    # N
                dest = self.get_n_coords(start)

            elif plane_index == 1:  # NE
                dest = self.get_ne_coords(start)

            elif plane_index == 2:  # SE
                dest = self.get_se_coords(start)

            elif plane_index == 3:  # S
                dest = self.get_s_coords(start)

            elif plane_index == 4:  # SW
                dest = self.get_sw_coords(start)

            elif plane_index == 5:  # NW
                dest = self.get_nw_coords(start)
            else:
                print("Problem parsing action...Exiting")
                exit()

        # NO_MOVE PLANE
        elif current_plane < (self.placement_planes + self.movement_planes + self.fight_planes + self.no_move_planes):
            act = 3
            start = (action_coords[1],action_coords[2])

        # NO_FIGHT PLANE
        elif current_plane < (self.placement_planes + self.movement_planes + self.fight_planes + self.no_move_planes + self.no_fight_planes):
            act = 4
            start = (action_coords[1],action_coords[2])

        else:
            print("Problem parsing action...Exiting")
            exit()

        return (act, start, dest)

    def play_action(self, action_coords):
        (act, start, dest) = self.parse_action(action_coords)

        if (act == 0): # placement
            player = self.current_player
            turn = self.current_turn
            new_unit = self.current_reinforcements[player-1][turn-1].pop(0)

            new_unit.move_to(*start, 0)
            self.available_units[self.current_player-1].append(new_unit)
            self.board[start[0]][start[1]].place_unit(new_unit)
            #print(self.available_units)

        elif (act == 1): # movement
            unit = self.board[start[0]][start[1]].unit

            if start != dest:
                start_tile = self.board[start[0]][start[1]]
                dest_tile = self.board[dest[0]][dest[1]]
                
                terrain = dest_tile.get_terrain()
                cost = terrain.cost

                unit.move_to(dest[0],dest[1], cost)
                start_tile.unit = None
                dest_tile.place_unit(unit)

                # Ends the movement of units who don't have enough movement points
                # to reduce total number of decisions that need to be taken per game
                if not any(self.check_mobility(unit, consider_other_units=False)): 
                    self.end_movement(unit)
            
            else:
                print("Problem playing action.\n \
                      Probably there is a bug in possible_actions().\n \
                      Exiting")
                exit()

        elif (act == 2): # fighting
            atacker_tile = self.board[start[0]][start[1]]
            defender_tile = self.board[dest[0]][dest[1]]
            self.resolve_combat(atacker_tile, defender_tile)

        elif (act == 3): # no movement
            unit = self.board[start[0]][start[1]].unit
            self.end_movement(unit)

        elif (act == 4): # no fighting
            unit = self.board[start[0]][start[1]].unit
            self.end_fighting(unit)

        else:
            print("Unknown action. Exiting...")
            exit()

        return

    # --------------- ENVIRONMENT --------------- #

    def reset_env(self):
        self.current_player = 1  
        self.current_phase = 0   
        self.current_stage = 0   
        self.current_turn = 1

        self.length = 0
        self.terminal_value = 0
        self.terminal = False

        for p in [0,1]:
            self.available_units[p].clear()
            self.moved_units[p].clear()
            self.atacked_units[p].clear()
        
        self.current_reinforcements = copy(self.all_reinforcements)

        
        for i in range(self.rows):
            for j in range(self.columns):
                self.board[i][j].reset() # reset each tile    

        
        # MCTS RELATED ATRIBUTES 
        self.child_policy.clear()
        self.state_history.clear()
        self.player_history.clear()
        self.action_history.clear()

        return
    
    def update_game_env(self):
        
        # Two players: 1 and 2
        # Three phases: placement, movement and fighting
        # Six stages: P1_Placement, P2_Placement, P1_Movement, P1_Fighting, P2_Movement, P2_Fighting

        done = False
        previous_player = self.current_player
        previous_stage = self.current_stage
        stage = previous_stage
        
        while True:
            if stage == 0 and self.current_reinforcements[0][self.current_turn-1] == []:     # first player used all his reinforcements
                stage+=1
                continue
            if stage == 1 and self.current_reinforcements[1][self.current_turn-1] == []:     # second player used all his reinforcements
                stage+=1
                continue
            if stage == 2 and self.available_units[0] == []:    # first player moved all his units
                stage+=1
                continue
            if stage == 3 and self.moved_units[0] == []:        # first player atacked with all his units
                stage+=1
                continue
            if stage == 4 and self.available_units[1] == []:    # second player moved all his units
                stage+=1
                continue
            if stage == 5 and self.moved_units[1] == []:        # second player atacked with all his units
                if self.current_turn+1 > self.turns:
                    done = True
                    self.terminal = True
                    break
                self.current_turn+=1
                stage=0
                self.new_turn()
                continue
            break
        
        if(done):
            self.terminal_value = self.check_terminal()
    
        # ------------------------------------    

        if stage in (0,2,3):
            self.current_player = 1
        elif stage in (1,4,5):
            self.current_player = 2
        else:
            print("Error in function: \'update_game_env()\'.Exiting")
            exit()

        if stage in (0,1):
            self.current_phase = 0
        elif stage in (2,4):
            self.current_phase = 1
        elif stage in (3,5):
            self.current_phase = 2
        else:
            print("Error in function: \'update_game_env()\'.Exiting")
            exit()

        self.current_stage = stage

        return done

    def new_turn(self):
        self.available_units = self.atacked_units.copy()
        self.atacked_units = [[], []]

        for p in [0,1]:
            for unit in self.available_units[p]:
                unit.reset_mov()

        return  

    def check_terminal(self):
        p1_captured_points = 0
        p2_captured_points = 0
        for point in self.victory_p1:
            vic_p1 = self.board[point[0]][point[1]]
            if vic_p1.unit:
                if(vic_p1.unit.player==2):
                    p2_captured_points +=1
        for point in self.victory_p2:
            vic_p2 = self.board[point[0]][point[1]]
            if vic_p2.unit:
                if(vic_p2.unit.player==1):
                    p1_captured_points +=1

        p1_percentage_captured = p1_captured_points / self.n_vp[1]
        p2_percentage_captured = p2_captured_points / self.n_vp[0]

        if p1_percentage_captured > p2_percentage_captured:
            final_value = 1     # p1 victory
        elif p1_percentage_captured < p2_percentage_captured:
            final_value = -1    # p2 victory
        else:
            final_value = 0     # draw
        
        return final_value

    def get_winner(self):
        terminal_value = self.get_terminal_value()

        if terminal_value < 0:
            winner = 2
        elif terminal_value > 0:
            winner = 1
        else:
            winner = 0

        return winner

    # ------------------ OTHER ------------------ #

    def resolve_combat(self, atacker_tile, defender_tile):
        atacker_unit = atacker_tile.unit
        defender_unit = defender_tile.unit
        atacking_player = atacker_unit.player
        defending_player = defender_unit.player

        defender_x = defender_unit.row
        defender_y = defender_unit.col
        
        if atacking_player == defending_player:
            print("\n\nSomething wrong with the attack. Player atacking itself!\nExiting")
            exit()

        defense_modifier = defender_tile.get_terrain().defense_modifier
        attack_modifier = atacker_tile.get_terrain().attack_modifier
        
        remaining_atacker_defense = atacker_unit.defense - (defender_unit.attack*defense_modifier)
        remaining_defender_defense = defender_unit.defense - (atacker_unit.attack*attack_modifier)
        

        if remaining_defender_defense <= 0:     # Defender died

            if remaining_atacker_defense<=0:   # Atacker died
                defender_tile.unit = None
                self.moved_units[atacking_player-1].remove(atacker_unit)

            else:                              # Atacker lived           
                atacker_unit.edit_defense(remaining_atacker_defense)

                # Take the enemy position
                defender_tile.unit = atacker_unit
                
                self.moved_units[atacking_player-1].remove(atacker_unit)
                atacker_unit.move_to(defender_x, defender_y, cost=0) # don't pay terrain cost when taking enemy position
                self.atacked_units[atacking_player-1].append(atacker_unit)

            # Acording to who controls the defending unit it will be in a different status
            if defending_player==1:
                self.atacked_units[defending_player-1].remove(defender_unit)

            elif defending_player==2:
                self.available_units[defending_player-1].remove(defender_unit)

            else:
                print("Problem with enemy player.Exiting.")
                exit()

            atacker_tile.unit = None

        else:                                   # Defender lived

            if remaining_atacker_defense<=0:   # Atacker died
                defender_unit.edit_defense(remaining_defender_defense)
                self.moved_units[atacking_player-1].remove(atacker_unit)
                atacker_tile.unit = None

            else:                               # Both lived
                atacker_unit.edit_defense(remaining_atacker_defense)
                defender_unit.edit_defense(remaining_defender_defense)
                self.end_fighting(atacker_unit)
        
        return

    def check_tiles(self, coords):
        # Clock-wise rotation order

        ''' 
             n
        nw   __   ne
            /  \ 
            \__/ 
        sw        se
              s
        '''
        (row, col) = coords

        n = None
        ne = None
        se = None
        s = None
        sw = None
        nw = None


        if (row-1) != -1:
            (x, y) = self.get_n_coords(coords)
            n = self.board[x][y]

        if (row+1) != self.rows:
            (x, y) = self.get_s_coords(coords)
            s = self.board[x][y]

        if not ((col == 0) or (row == 0 and col % 2 == 0)):
            (x, y) = self.get_nw_coords(coords)
            nw = self.board[x][y]

        if not ((col == 0) or (row == self.rows-1 and col % 2 != 0)):
            (x, y) = self.get_sw_coords(coords)
            sw = self.board[x][y]

        if not ((col == self.columns-1) or (row == 0 and col % 2 == 0)):
            (x, y) = self.get_ne_coords(coords)
            ne = self.board[x][y]

        if not ((col == self.columns-1) or (row == self.rows-1 and col % 2 != 0)):
            (x, y) = self.get_se_coords(coords)
            se = self.board[x][y]
        

        return n, ne, se, s, sw, nw

    def check_mobility(self, unit, consider_other_units=False):
        x = unit.row
        y = unit.col
        
        tiles = self.check_tiles((x,y))

        can_move = [False for i in range(len(tiles))]

        for i in range(len(tiles)):
            tile = tiles[i]
            if tile:
                cost = tile.get_terrain().cost
                dif = unit.mov_points - cost
                if dif >= 0:
                    can_move[i] = True
                    if consider_other_units and tile.unit:
                        can_move[i] = False

        return can_move
    
    def check_adjacent_units(self, unit, enemy):
        x = unit.row
        y = unit.col
        
        tiles = self.check_tiles((x,y))
            
        adjacent_units = []
        enemy_directions = []

        for i in range(len(tiles)):
            tile = tiles[i]
            if tile:
                unit = tile.unit
                if unit:
                    if (unit.player==enemy):
                        adjacent_units.append(unit)
                        enemy_directions.append(i)

        return adjacent_units , enemy_directions

    def end_movement(self, unit):
        player = unit.player
        enemy = (not(player-1)) + 1

        # End movement
        self.moved_units[unit.player-1].append(unit)
        self.available_units[unit.player-1].remove(unit)

        # Marks units without adjancent enemies as done fighting
        # to reduce the total number of decisions that need to be taken
        _, enemy_directions = self.check_adjacent_units(unit, enemy)
        if len(enemy_directions) == 0:
            self.end_fighting(unit)

    def end_fighting(self, unit):
        self.atacked_units[unit.player-1].append(unit)
        self.moved_units[unit.player-1].remove(unit)
    
##########################################################################
# -------------------------                   -------------------------- #
# ------------------------   UTILITY METHODS   ------------------------- #
# -------------------------                   -------------------------- #
##########################################################################

    def get_n_coords(self, coords):
        (row, col) = coords
        n = (row-1, col)
        return n
    
    def get_ne_coords(self, coords):
        (row, col) = coords
        if col % 2 == 0:
            ne = (row-1, col+1)
        else:
            ne = (row, col+1)

        return ne
    
    def get_se_coords(self, coords):
        (row, col) = coords
        if col % 2 == 0:
            se = (row, col+1)
        else:
            se = (row+1, col+1)
    
        return se
    
    def get_s_coords(self, coords):
        (row, col) = coords
        s = (row+1, col)
        return s

    def get_sw_coords(self, coords):
        (row, col) = coords
        if col % 2 == 0:
            sw = (row, col-1)
        else:
            sw = (row+1, col-1)

        return sw
    
    def get_nw_coords(self, coords):
        (row, col) = coords
        if col % 2 == 0:
            nw = (row-1, col-1)
        else:
            nw = (row, col-1)

        return nw

    def define_board_sides(self):

        # Calculate the indexes that define each side of the board
        if self.columns % 2 != 0:
            middle_index = math.floor(self.columns/2)
            self.p1_last_index = middle_index-1
            self.p2_first_index = middle_index+1
        else:
            # if number of rows is even there are two middle collumns one on the right and one on the left
            mid = int(self.columns/2)
            left_side_collumn = mid
            right_side_collumn = mid + 1
            left_index = left_side_collumn - 1
            right_index = right_side_collumn - 1
            
            # For boards with even columns we separate the center by one more column
            self.p1_last_index = max(0, left_index-1)
            self.p2_first_index = min(self.columns-1, right_index+1)

##########################################################################
# -------------------------                   -------------------------- #
# ------------------------  ALPHAZERO SUPPORT  ------------------------- #
# -------------------------                   -------------------------- #
##########################################################################

    def generate_state_image(self):
        # Terrain Channels #
        atack_modifiers = torch.ones((self.rows, self.columns))
        defense_modifiers = torch.ones((self.rows, self.columns))
        movement_costs = torch.ones((self.rows, self.columns))

        for i in range(self.rows):
            for j in range(self.columns):
                tile = self.board[i][j]
                terrain = tile.get_terrain()
                a = terrain.attack_modifier
                d = terrain.defense_modifier
                m = terrain.cost
                atack_modifiers[i][j] = a
                defense_modifiers[i][j] = d
                movement_costs[i][j] = m

        atack_modifiers = torch.unsqueeze(atack_modifiers, 0)
        defense_modifiers = torch.unsqueeze(defense_modifiers, 0)
        movement_costs = torch.unsqueeze(movement_costs, 0)


        # Reinforcements Channels #
        player_reinforcements = [None, None]

        for player, reinforcements in self.current_reinforcements.items():
            for turn in range(self.n_reinforcement_turns):
                if (self.current_turn + turn) < self.turns:
                    turn_index = (self.current_turn + turn)-1
                    turn_reinforcements = reinforcements[turn_index]
                else:
                    turn_reinforcements = []

                for r in range(self.max_reinforcements):
                    if r < len(turn_reinforcements):
                        unit = turn_reinforcements[r]
                        unit_stats = [unit.attack, unit.defense, unit.mov_points]
                        unit_planes = None
                        for stat in unit_stats:
                            # Currently reinforcements can be placed on each player's side of the board
                            available_columns = self.p1_last_index + 1 # number of columns on my side of the board
                            my_half = torch.full((self.rows, available_columns), stat)
                            rest_of_columns = self.columns - available_columns
                            enemy_half = torch.zeros((self.rows, rest_of_columns))       
                            if player == 1:
                                stat_plane = torch.unsqueeze(torch.cat((my_half, enemy_half), axis=1), 0)
                            else:
                                stat_plane = torch.unsqueeze(torch.cat((enemy_half, my_half), axis=1), 0)

                            if unit_planes is None:
                                unit_planes = stat_plane
                            else:
                                unit_planes = torch.cat((unit_planes, stat_plane), dim=0)
                    else:
                        unit_planes = torch.zeros((self.N_UNIT_STATS, self.rows, self.columns))

                    if player_reinforcements[player] is None:
                        player_reinforcements[player] = unit_planes
                    else:
                        player_reinforcements[player] = torch.cat((player_reinforcements[player], unit_planes))
                    

        p1_reinforcements = player_reinforcements[0]
        p2_reinforcements = player_reinforcements[1]
                    

        # Victory Points Channels #
        p1_victory = np.zeros((self.rows, self.columns), dtype=np.int32)
        p2_victory = np.zeros((self.rows, self.columns), dtype=np.int32)

        for v in self.victory_p1:
            x = v[0]
            y = v[1]
            p1_victory[x][y] = 1

        for v in self.victory_p2:
            x = v[0]
            y = v[1]
            p2_victory[x][y] = 1

        p1_victory = torch.unsqueeze(torch.as_tensor(p1_victory, dtype=torch.float32), 0)
        p2_victory = torch.unsqueeze(torch.as_tensor(p2_victory, dtype=torch.float32), 0)


        # Unit Representation Channels #
        p1_units = torch.zeros(self.N_UNIT_STATS * self.N_UNIT_STATUSES, self.rows, self.columns)
        p2_units = torch.zeros(self.N_UNIT_STATS * self.N_UNIT_STATUSES, self.rows, self.columns)
        p_units = [p1_units, p2_units]
        for p in [0,1]: 
            # for each player check each unit status
            statuses_list = [self.available_units[p], self.moved_units[p], self.atacked_units[p]]
            for status_index in range(len(statuses_list)):
                unit_list = statuses_list[status_index]
                for unit in unit_list:
                    first_index = status_index * self.N_UNIT_STATS
                    row = unit.row
                    col = unit.col
                    p_units[p][first_index + 0][row][col] = unit.attack
                    p_units[p][first_index + 1][row][col] = unit.defense
                    p_units[p][first_index + 2][row][col] = unit.mov_points
            

        # Player Channel #
        player_plane = np.ones((self.rows,self.columns), dtype=np.int32)
        if self.current_player == 2:
            player_plane.fill(-1)

        player_plane = torch.unsqueeze(torch.as_tensor(player_plane,dtype=torch.float32), 0)

        # Phase Channel #
        phase = self.current_phase
        phase_plane = torch.full((self.rows, self.columns), phase, dtype=torch.float32)
        phase_plane = torch.unsqueeze(phase_plane, 0)

        # Turn Channel #
        turn_percent = self.current_turn/self.turns
        turn_plane = torch.full((self.rows, self.columns), turn_percent, dtype=torch.float32)
        turn_plane = torch.unsqueeze(turn_plane, 0)

        # Final operations #
        stack_list = []

        terrain_list = [atack_modifiers, defense_modifiers, movement_costs]
        stack_list.extend(terrain_list)
    
        core_list = [p1_victory, p2_victory, p1_units, p2_units, p1_reinforcements, p2_reinforcements, phase_plane, turn_plane, player_plane]
        stack_list.extend(core_list)
        new_state = torch.concat(stack_list, dim=0)

        state_image = torch.unsqueeze(new_state, 0) # add batch size to the dimensions

        #print(state_image)
        return state_image
    
    def store_search_statistics(self, node):
        sum_visits = sum(child.visit_count for child in node.children.values())
        self.child_policy.append(
            [ node.children[a].visit_count / sum_visits if a in node.children else 0
            for a in range(self.num_actions) ])

    def make_target(self, i):
        value_target = self.terminal_value
        policy_target = self.child_policy[i]

        target = (value_target, policy_target)
        return target

    def debug_state_image(self, state_image):
        for channel in state_image[0]:
            print(channel)

##########################################################################
# -------------------------                    ------------------------- #
# ------------------------   INSTANCE METHODS   ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def load_from_config(self, filename):
        # Read YAML file
        with open(filename, 'r') as stream:
            data_loaded = yaml.safe_load(stream)

        units_by_id = {}
        terrain_by_id = {}
        all_reinforcements = {}
        
        for section_name, values in data_loaded.items():
            match section_name:
                case "Board_dimensions":
                    self.rows = values["rows"]
                    self.columns = values["columns"]
                    self.define_board_sides()

                case "Turns":
                    self.turns = values

                case "Units":
                    for unit_name, properties in values.items():
                        id = properties["id"]
                        properties.pop("id")
                        units_by_id[id] = {}
                        units_by_id[id]["name"] = unit_name
                        units_by_id[id].update(properties)
            
                case "Reinforcements":
                    max_reinforcements = 0
                    for p, reinforcements in values.items():
                        player_index = int(p[-1]) - 1

                        all_reinforcements[player_index] = []
                        for turn_idx in range(len(reinforcements)):
                            turn_units = reinforcements[turn_idx]
                            n_units = len(turn_units)
                            if n_units > max_reinforcements:
                                max_reinforcements = n_units

                            all_reinforcements[player_index].append([])
                            for id in turn_units:
                                player = player_index + 1
                                name = units_by_id[id]["name"]
                                attack = units_by_id[id]["attack"]
                                defense = units_by_id[id]["defense"]
                                mov_allowance = units_by_id[id]["movement"]
                                image_path = units_by_id[id]["image_path"]
                                all_reinforcements[player_index][turn_idx].append(Unit(name, attack, defense, mov_allowance, player, image_path))

                    self.all_reinforcements = all_reinforcements
                    self.current_reinforcements = self.all_reinforcements.copy()
                    self.max_reinforcements = max_reinforcements

                case "Terrain":
                    for terrain_name, properties in values.items():
                        id = properties["id"]
                        properties.pop("id")
                        terrain_by_id[id] = {}
                        terrain_by_id[id]["name"] = terrain_name
                        terrain_by_id[id].update(properties)

                case "Map":
                    self.terrain_types = []
                    for id, properties in terrain_by_id.items():
                        instance =  Terrain(
                            attack_modifier=properties["attack_modifier"],
                            defense_modifier=properties["defense_modifier"],
                            cost=properties["cost"], 
                            name=properties["name"],
                            image_path=properties["image_path"])
                        
                        properties["instance"] = instance
                        self.terrain_types.append(instance)

                    method = values["creation_method"]
                    if method == "Randomized":
                        if values.get("distribution"):
                            distribution = values["distribution"]
                        else:
                            num_terrains = len(terrain_by_id)
                            distribution = [1/num_terrains for _ in range(num_terrains)]

                        for i in range(self.rows):
                            self.board.append([])
                            for j in range(self.columns): 
                                terrain = np.random.choice(self.terrain_types, p=distribution)
                                self.board[i].append(Tile(i,j,terrain))

                    elif method == "Detailed":
                        map_configuration = values["map_configuration"]
                        map_shape = np.shape(map_configuration)
                        if map_shape != (self.rows, self.columns):
                            print("Wrong shape for map configuration. Exiting")
                            exit()
                        else:
                            for i in range(self.rows):
                                self.board.append([])
                                for j in range(self.columns):
                                    terrain_id = map_configuration[i][j]
                                    terrain = terrain_by_id[terrain_id]["instance"]
                                    self.board[i].append(Tile(i,j,terrain))
                    else:
                        print("Unrecognized map creation method. Exiting")
                        exit()

                case "Victory_points":
                    method = values["creation_method"]

                    if method == "Randomized":
                        p1_vp = values["number_vp"]["p1"]
                        p2_vp = values["number_vp"]["p2"]
                        self.victory_p1 = []        
                        self.victory_p2 = []

                        p1_available_tiles = self.rows * self.p1_last_index+1
                        p2_available_tiles = self.rows * self.p1_last_index+1
                        if p1_vp > p1_available_tiles:
                            print("Game config has too many victory points for p1.")
                            exit()
                        if p2_vp > p2_available_tiles:
                            print("Game config has too many victory points for p2.")
                            exit()

                        for _ in range(p1_vp):
                            row = np.random.choice(range(self.rows))
                            col = np.random.choice(range(self.p1_last_index+1))
                            point = (row, col)
                            while point in self.victory_p1:
                                row = np.random.choice(range(self.rows))
                                col = np.random.choice(range(self.p1_last_index+1))
                                point = (row, col)

                            self.victory_p1.append(point)
                        
                        for _ in range(p2_vp):
                            row = np.random.choice(range(self.rows))
                            col = np.random.choice(range(self.p2_first_index, self.columns))
                            point = (row, col)
                            while point in self.victory_p2:
                                row = np.random.choice(range(self.rows))
                                col = np.random.choice(range(self.p2_first_index, self.columns))
                                point = (row, col)

                            self.victory_p2.append(point)


                    elif method == "Detailed":
                        p1_vp = values["vp_locations"]["p1"]
                        p2_vp = values["vp_locations"]["p2"]
                        self.victory_p1 = []        
                        self.victory_p2 = []

                        loaded_vps = [p1_vp, p2_vp]
                        game_vps = [self.victory_p1, self.victory_p2]
                        for player in loaded_vps:
                            loaded_list = loaded_vps[player]
                            game_list = game_vps[player]
                            for point in loaded_list:
                                if len(point) != 2:
                                    print(str(point) + " --> Points must have two coordenates.")
                                    exit()
                                elif point in game_list:
                                    print(str(point) + " --> Repeated point. Cannot have two points with the same coordenates.")
                                    exit()
                                else:
                                    vp_tuple = (point[0], point[1])
                                    game_list.append(vp_tuple)        

                    else:
                        print("Unrecognized victory points creation method. Exiting")
                        exit()

                    self.n_vp = [0, 0]
                    for point in self.victory_p1:
                        self.board[point[0]][point[1]].victory = 1
                        self.n_vp[0] += 1

                    for point in self.victory_p2:
                        self.board[point[0]][point[1]].victory = 2
                        self.n_vp[1] += 1

    def clone(self):
        return deepcopy(self)
    
    def shallow_clone(self):
        ignore_list = ["child_policy", "state_history", "player_history", "action_history"]
        new_game = SCS_Game()

        memo = {} # memo dict for deepcopy so that it knows what objects it has already copied before
        attributes = self.__dict__.items()
        for name, value in attributes:
            if (not(name.startswith('__') and name.endswith('__'))) and (name not in ignore_list):
                value_copy = deepcopy(value, memo)
                setattr(new_game, name, value_copy)
                
        return new_game

##########################################################################
# ----------------------                         ----------------------- #
# ----------------------  REPRESENTATION METHODS  ---------------------- #
# ----------------------                         ----------------------- #
##########################################################################

    def string_representation(self):
        print("string representation for squared boards")

        string = "\n   "
        for k in range(self.columns):
            string += (" " + format(k+1, '02') + " ")
        
        string += "\n  |"
        for k in range(self.columns-1):
            string += "---|"

        string += "---|\n"

        for i in range(self.rows):
            string += format(i+1, '02') + "| "
            for j in range(self.columns):
                mark = " "
                if self.board[i][j].victory == 1:
                    mark = colored("V", "cyan")
                elif self.board[i][j].victory == 2:
                    mark = colored("V", "yellow")

                unit = self.board[i][j].unit
                if unit:    
                    if unit.player == 1:
                        mark=colored("U", "blue")
                    else:
                        mark=colored("U", "red")

                string += mark + ' | '
            string += "\n"

            if(i<self.rows-1):
                string += "  |"
                for k in range(self.columns-1):
                    string += "---|"
                string += "---|\n"
            else:
                string += "   "
                for k in range(self.columns-1):
                    string += "--- "
                string += "--- \n"

        string += "=="
        for k in range(self.columns):
            string += "===="
        string += "==\n"

        return string

    def string_action(self, action_coords):

        parsed_action = self.parse_action(action_coords)

        act = parsed_action[0]
        start = parsed_action[2]
        dest = parsed_action[3]

        string = ""
        if (act == 0): # placement
            string = "Unit placement phase: Placing a Unit "
        
            string = string + "at (" + str(start[0]+1) + "," + str(start[1]+1) + ")"

        elif (act == 1): # movement
            string = "Movement phase: Moving from (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "to (" + str(dest[0]+1) + "," + str(dest[1]+1) + ")"
        
        elif (act == 2): # fighting
            string = "Fighting phase: Using unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "to atack unit at (" + str(dest[0]+1) + "," + str(dest[1]+1) + ")"

        elif (act == 3): # no move
            string = "Movement phase: Unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "chose not to move"

        elif (act == 4): # no fight
            string = "Fighting phase: Unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "chose not to fight"

        else:
            string = "Unknown action..."

        #print(string)
        return string