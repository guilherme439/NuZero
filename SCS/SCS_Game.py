import copy
import math
import numpy as np
import torch
import enum
import sys
import gc
import time
import hexagdly

from enum import Enum
from collections import Counter
from termcolor import colored

from .Tile import Tile
from .Soldier import Soldier
from .Tank import Tank
from .Terrain import Terrain



class SCS_Game():

    WIDTH = 5
    HEIGHT = 5
    TURNS = 7

    N_PLAYERS = 2

    N_VP = 1            # Number of victory points
    N_UNIT_TYPES = 2
    N_UNIT_STATUSES = 3



    def __init__(self, r1=[1,1], r2=[1,1], use_terrain=True):

        # ------------------------------------------------------------ #
        # ---------------------- INITIALIZATION ---------------------- #
        # ------------------------------------------------------------ #

        self.board = []
        self.current_player = 1
        self.current_phase = 0
        self.current_stage = 0   
        self.current_turn = 1
        self.reinforcements = [[],[]]
        self.available_units = [[],[]]
        self.moved_units = [[],[]]
        self.atacked_units = [[],[]]

        self.use_terrain = use_terrain
        self.terrain_types = []

        self.victory_p1 = [ [-1,-1] for _ in range(self.N_VP) ]
        self.victory_p2 = [ [-1,-1] for _ in range(self.N_VP) ]

        self.reinforcements_by_type = [[],[]]
        self.reinforcements_by_type[0] = r1 # number of soldiers , number of tanks... etc
        self.reinforcements_by_type[1] = r2

        self.reinforcements_as_list = [[],[]]

        for p in range(self.N_PLAYERS):
            self.reinforcements_as_list[p] = np.concatenate(
            [
            np.repeat(i+1, self.reinforcements_by_type[p][i])
            for i in range(len(self.reinforcements_by_type[p]))
            ]
            ).tolist()       


        # ------------------------------------------------------------ #
        # ----------------------- BOARD SETUP ------------------------ #
        # ------------------------------------------------------------ #

        ## TILES AND TERRAIN
        map_choice = 1
        if not self.use_terrain:

            for i in range(self.HEIGHT):
                self.board.append([])
                for j in range(self.WIDTH):
                    self.board[i].append(Tile(i,j))

        else:
            if map_choice == 1:
            # Distribution based map

                mountain = Terrain(atack_modifier=1, defense_modifier=2, cost=2, name="Mountain", image_path="SCS/Images/mountain.jpg")
                plains = Terrain(atack_modifier=1, defense_modifier=1, cost=1, name="Plains", image_path="SCS/Images/plains.jpg")
                bush = Terrain(atack_modifier=2, defense_modifier=1/2, cost=1, name="Bush", image_path="SCS/Images/plains_with_bush.jpg")
                swamp = Terrain(atack_modifier=1/2, defense_modifier=1/2, cost=2, name="Swamp", image_path="SCS/Images/swamp.jpg")

                self.terrain_types = [mountain, plains, bush, swamp]
                probs = [0.1, 0.65, 0.15, 0.1]

                for i in range(self.HEIGHT):
                    self.board.append([])
                    for j in range(self.WIDTH): 
                        terrain = np.random.choice(self.terrain_types, p=probs)
                        self.board[i].append(Tile(i,j,terrain))


        ## VICTORY POINTS

        # p1 victory points will be on the left side and p2 victory points on the right side
        if self.WIDTH % 2 != 0:
            middle_index = math.floor(self.WIDTH/2)
            self.p1_last_index = middle_index-1
            self.p2_first_index = middle_index+1
        else:
            # if number of rows is even there are two middle collumns one on the right and one on the left
            mid = int(self.WIDTH/2)
            left_side_collumn = mid
            right_side_collumn = mid + 1
            left_index = left_side_collumn - 1
            right_index = right_side_collumn - 1
            
            # so that victory points are not adjacent, we separate them by one more collumn
            self.p1_last_index = max(0, left_index-1)
            self.p2_first_index = min(self.WIDTH-1, right_index+1)

        x_coords_p1 = np.random.choice(range(self.HEIGHT), size=self.N_VP, replace=False)
        y_coords_p1 = np.random.choice(range(self.p1_last_index+1), size=self.N_VP, replace=False)

        for i in range(len(self.victory_p1)):
            self.victory_p1[i][0]=x_coords_p1[i]
            self.victory_p1[i][1]=y_coords_p1[i]
                
        x_coords_p2 = np.random.choice(range(self.HEIGHT), size=self.N_VP, replace=False)
        y_coords_p2 = np.random.choice(range(self.p2_first_index, self.WIDTH), size=self.N_VP, replace=False)        

        for i in range(len(self.victory_p2)):
            self.victory_p2[i][0]=x_coords_p2[i]
            self.victory_p2[i][1]=y_coords_p2[i]
    
        for point in self.victory_p1:
            self.board[point[0]][point[1]].victory = 1

        for point in self.victory_p2:
            self.board[point[0]][point[1]].victory = 2


        ## INITIAL REINFORCEMENTS
        for p in range(self.N_PLAYERS):
            self.reinforcements[p] = copy.copy(self.reinforcements_as_list[p])
        

        # ------------------------------------------------------------ #
        # ------------- SPACE AND ACTION REPRESENTATION -------------- #
        # ------------------------------------------------------------ #

        ## ACTION REPRESENTATION

        # Number of planes per type
        self.placement_planes = self.N_UNIT_TYPES 
        self.movement_planes = 4 # {N, S, E, 0} directions
        self.fight_planes = 4 # {N, S, E, 0} directions
        self.no_move_planes = 1
        self.no_fight_planes = 1

        self.total_action_planes = \
        self.placement_planes + \
        self.movement_planes + \
        self.fight_planes + \
        self.no_move_planes + \
        self.no_fight_planes

        self.action_space_shape = (self.total_action_planes , self.HEIGHT , self.WIDTH)
        self.num_actions     =     self.total_action_planes * self.HEIGHT * self.WIDTH


        ## STATE REPRESENTATION
        n_vic_dims = self.N_PLAYERS
        n_reinforcement_dims = self.N_PLAYERS * self.N_UNIT_TYPES
        n_unit_dims = self.N_PLAYERS * self.N_UNIT_TYPES * self.N_UNIT_STATUSES
        n_feature_dims = 3 # turn, phase and player
        total_dims = n_vic_dims + n_reinforcement_dims + n_unit_dims + n_feature_dims

        n_terrain_dims = 3 # atack, defense, movement
        if self.use_terrain:
            total_dims += n_terrain_dims

        self.game_state_shape = (total_dims, self.HEIGHT, self.WIDTH)


        # ------------------------------------------------------ #
        # --------------- MCTS RELATED ATRIBUTES --------------- #
        # ------------------------------------------------------ #

        self.terminal = False
        self.length = 0
        self.terminal_value = 0
        self.child_policy = []
        self.state_history = []
        self.player_history = []
        self.action_history = []


        self.update_game_env() # Just in case
        return
    
##########################################################################
# -------------------------                    ------------------------- #
# -----------------------  GET AND SET METHODS  ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def get_board(self):
        return self.board
    
    def getBoardWidth(self):
        return self.WIDTH

    def getBoardHeight(self):
        return self.HEIGHT    

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
    
    def get_name(self):
        return "SCS"
    
##########################################################################
# ----------------------------              ---------------------------- #
# --------------------------  CORE FUNCTIONS  -------------------------- #
# ----------------------------              ---------------------------- #
##########################################################################

    def reset_env(self):

        self.current_player = 1  
        self.current_phase = 0   
        self.current_stage = 0   
        self.current_turn = 1

        for p in [0,1]:
            self.available_units[p].clear()
            self.moved_units[p].clear()
            self.atacked_units[p].clear()
            self.reinforcements[p].clear()
            self.reinforcements[p] = copy.copy(self.reinforcements_as_list[p])

        
        for i in range(self.HEIGHT):
            for j in range(self.WIDTH):
                self.board[i][j].reset() # reset each tile    

        
        # MCTS RELATED ATRIBUTES 
        self.terminal = False
        self.terminal_value = 0
        self.length = 0
        self.child_policy.clear()
        self.state_history.clear()
        self.player_history.clear()
        self.action_history.clear()

        return
    
    def possible_actions(self):
        player = self.current_player
        phase = self.current_phase
        size = self.HEIGHT * self.WIDTH
        
        placement_planes = []
        movement_planes = np.zeros((4, self.HEIGHT, self.WIDTH), dtype=np.int32)
        fight_planes = np.zeros((4, self.HEIGHT, self.WIDTH), dtype=np.int32)
        no_move_plane = np.zeros((1, self.HEIGHT, self.WIDTH), dtype=np.int32)
        no_fight_plane = np.zeros((1, self.HEIGHT, self.WIDTH), dtype=np.int32)
        
        #print(phase)
        if (phase == 0):
            available_types = set(self.reinforcements[player-1])
            for t in range(self.N_UNIT_TYPES):
                if t+1 in available_types:
                    available_collumns = self.p1_last_index + 1 # number of columns on my side of the board
                    my_half = np.ones((self.HEIGHT, available_collumns), dtype=np.int32)
                    rest_of_columns = self.WIDTH - available_collumns
                    enemy_half = np.zeros((self.HEIGHT, rest_of_columns), dtype=np.int32)
                        
                    if player == 1:
                        u_plane = np.concatenate((my_half, enemy_half), axis=1)
                    else:
                        u_plane = np.concatenate((enemy_half, my_half), axis=1)
                else:
                    u_plane = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)


                placement_planes.insert(t,u_plane)
            

            for p in [0,1]:
                for unit in self.available_units[p]:
                    x = unit.tile_x
                    y = unit.tile_y
                    for t in range(self.N_UNIT_TYPES):
                        placement_planes[t][x][y] = 0
            # can not place on top of other units
            
        
        if (phase == 1):
            placement_planes = np.zeros((self.N_UNIT_TYPES, self.HEIGHT, self.WIDTH), dtype=np.int32)
            
            for unit in self.available_units[player-1]:
                x = unit.tile_x
                y = unit.tile_y

                no_move_plane[0][x][y] = 1 # no move action

                tiles = self.check_tiles((x,y))
                movements = self.check_mobility(unit, consider_other_units=True)

                for i in range(len(tiles)):
                    tile = tiles[i]
                    if (tile):
                        if(movements[i]):
                            movement_planes[i][x][y] = 1
                 
        if (phase == 2):
            placement_planes = np.zeros((self.N_UNIT_TYPES, self.HEIGHT, self.WIDTH), dtype=np.int32)

            for unit in self.moved_units[player-1]:
                pos_x = unit.tile_x
                pos_y = unit.tile_y
                
                no_fight_plane[0][pos_x][pos_y] = 1 # no fight action

                enemy_player = (not(player-1)) + 1 
                _ , enemy_dir = self.check_adjacent_units(unit, enemy_player)

                for direction in enemy_dir:
                    fight_planes[direction][pos_x][pos_y] = 1


        planes_list = [placement_planes, movement_planes, fight_planes, no_move_plane, no_fight_plane]
        valid_actions_mask = np.concatenate(planes_list)
        return valid_actions_mask

    def parse_action(self, action_coords):
        act = None           # Represents the type of action
        unit_type = None
        start = (None, None) # Starting point of the action
        dest = (None, None)  # Destination point for the action

        board_size = self.HEIGHT * self.WIDTH

        current_plane = action_coords[0]

        # PLACEMENT PLANES
        if current_plane < self.placement_planes:
            act = 0
            unit_type = action_coords[0] + 1  # 1 -> soldier | 2 -> tank
            start = (action_coords[1], action_coords[2])

        # MOVEMENT PLANES
        elif current_plane < (self.placement_planes + self.movement_planes):
            act = 1
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

            plane_index = current_plane-self.placement_planes
            # UP(0) DOWN(1) RIGHT(2) LEFT(3)
            if plane_index == 0:
                dest= (x-1,y)
            elif plane_index == 1:
                dest= (x+1,y)
            elif plane_index == 2:
                dest= (x,y+1)
            elif plane_index == 3:
                dest= (x,y-1)
            else:
                print("Problem parsing action...Exiting")
                exit()
            
        # FIGHT PLANES
        elif current_plane < (self.placement_planes + self.movement_planes + self.fight_planes):
            act = 2
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

            plane_index = current_plane - (self.placement_planes + self.movement_planes)

            # UP(0) DOWN(1) RIGHT(2) LEFT(3)
            if plane_index == 0:
                dest= (x-1,y)
            elif plane_index == 1:
                dest= (x+1,y)
            elif plane_index == 2:
                dest= (x,y+1)
            elif plane_index == 3:
                dest= (x,y-1)
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

        return (act, unit_type, start, dest)

    def play_action(self, action_coords):
        (act, unit_type, start, dest) = self.parse_action(action_coords)

        if (act == 0): # placement
            if unit_type == 1: 
                new_unit = Soldier(self.current_player,start[0],start[1])
                self.reinforcements[self.current_player-1].remove(1)
            elif unit_type == 2:
                new_unit = Tank(self.current_player,start[0],start[1])
                self.reinforcements[self.current_player-1].remove(2)
            else:
                print("Error playing action")
                exit()
            
            self.available_units[self.current_player-1].append(new_unit)
            self.board[start[0]][start[1]].place_unit(new_unit)

        elif (act == 1): # movement
            unit = self.board[start[0]][start[1]].unit

            if start != dest:
                start_tile = self.board[start[0]][start[1]]
                dest_tile = self.board[dest[0]][dest[1]]

                cost = 1
                if self.use_terrain:
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
                print("Bug in possible_actions(). Exiting")
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

    def update_game_env(self):
        
        # Two players: 1 and 2
        # Three phases: placement, movement and fighting
        # Six stages: P1_Placement, P2_Placement, P1_Movement, P1_Fighting, P2_Movement, P2_Fighting

        done = False
        previous_player = self.current_player
        previous_stage = self.current_stage
        stage = previous_stage
        
        while True:
            if stage == 0 and self.reinforcements[0] == []:     # first player used all his reinforcements
                stage+=1
                continue
            if stage == 1 and self.reinforcements[1] == []:     # second player used all his reinforcements
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
                if self.current_turn+1 > self.TURNS:
                    done = True
                    self.terminal = True
                    break
                self.current_turn+=1
                stage=0
                self.new_turn()
                continue
            break

        if(done):
            self.terminal_value = self.check_final_result()
    
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

    def step_function(self, action_coords):
        self.play_action(action_coords)
        self.action_history.append(action_coords)
        self.length += 1
        done = self.update_game_env()
        return done
    
    def check_tiles(self, coords):
        
        (x,y) = coords

        if (x-1) == -1:
            up = None
        else:
            up = self.board[x-1][y]

        if (x+1) == self.HEIGHT:
            down = None
        else:
            down = self.board[x+1][y]

        if (y+1) == self.WIDTH:
            right = None
        else:
            right = self.board[x][y+1]
        
        if (y-1) == -1:
            left = None
        else:
            left = self.board[x][y-1]

        return up, down, right, left

    def check_mobility(self, unit, consider_other_units=False):
        x = unit.tile_x
        y = unit.tile_y
        
        tiles = self.check_tiles((x,y))

        can_move = [False, False, False, False]

        for i in range(len(tiles)):
            tile = tiles[i]
            if tile:
                cost = 1
                if self.use_terrain:
                    cost = tile.get_terrain().cost
                dif = unit.mov_points - cost
                if dif >= 0:
                    can_move[i] = True
                    if consider_other_units and tile.unit:
                        can_move[i] = False

        return can_move
    
    def check_adjacent_units(self, unit, enemy):
        x = unit.tile_x
        y = unit.tile_y
        
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

    def resolve_combat(self, atacker_tile, defender_tile):
        atacker_unit = atacker_tile.unit
        defender_unit = defender_tile.unit
        atacking_player = atacker_unit.player
        defending_player = defender_unit.player

        defender_x = defender_unit.tile_x
        defender_y = defender_unit.tile_y
        
        if atacking_player == defending_player:
            print("\n\nSomething wrong with the atack. Player atacking itself!\nExiting")
            exit()

        if self.use_terrain:
            defense_modifier = defender_tile.get_terrain().defense_modifier
            atack_modifier = atacker_tile.get_terrain().atack_modifier
        else:
            defense_modifier = 1
            atack_modifier = 1

        remaining_atacker_defense = atacker_unit.defense - (defender_unit.atack*defense_modifier)
        remaining_defender_defense = defender_unit.defense - (atacker_unit.atack*atack_modifier)
        

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
            
    def check_final_result(self):

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

        point_diff = p1_captured_points - p2_captured_points
        p1_advantage = point_diff/self.N_VP

        if p1_advantage > 0:
            final_value = 1     # p1 victory
        elif p1_advantage < 0:
            final_value = -1    # p2 victory
        else:
            final_value = 0     # draw
        
        return final_value
    
    def check_victory(self, show=False):
        print("Warning: check victory was not updated and might be incorrect")

        winner = 0
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
        if p1_captured_points > p2_captured_points:
            if(show):
                print("PLAYER 1 WINS!")
            winner=1
        elif p2_captured_points > p1_captured_points:
            if(show):
                print("PLAYER 2 WINS!")
            winner=2
        elif p1_captured_points == 0:
            if(show):
                print("GAME DRAWN! :(")
        else:
            if(show):
                print("GAME DRAWN! :)")
            
        return winner

    def check_winner(self):
        terminal_value = self.get_terminal_value()

        if terminal_value < 0:
            winner = 2
        elif terminal_value > 0:
            winner = 1
        else:
            winner = 0

        return winner

    def generate_state_image(self):
        
        # Initialization
        p1_victory = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)
        p2_victory = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)

        p_units = [[],[]]

        p1_reinforcements = [[] for u in range(self.N_UNIT_TYPES)]
        p2_reinforcements = [[] for u in range(self.N_UNIT_TYPES)]

        p1_reinforcement_counts = list(Counter(self.reinforcements[0]).values())
        p2_reinforcement_counts = list(Counter(self.reinforcements[1]).values())

        
        # Terrain Channels
        if self.use_terrain:
            atack_modifiers = torch.ones((self.HEIGHT, self.WIDTH))
            defense_modifiers = torch.ones((self.HEIGHT, self.WIDTH))
            movement_costs = torch.ones((self.HEIGHT, self.WIDTH))

            for i in range(self.HEIGHT):
                for j in range(self.WIDTH):
                    tile = self.board[i][j]
                    terrain = tile.get_terrain()
                    a = terrain.atack_modifier
                    d = terrain.defense_modifier
                    m = terrain.cost
                    atack_modifiers[i][j] = a
                    defense_modifiers[i][j] = d
                    movement_costs[i][j] = m

            atack_modifiers = torch.unsqueeze(atack_modifiers, 0)
            defense_modifiers = torch.unsqueeze(defense_modifiers, 0)
            movement_costs = torch.unsqueeze(movement_costs, 0)



        # Reinforcements Channels
        for u in range(self.N_UNIT_TYPES):
            p1_value = 0
            p2_value = 0
            if u < len(p1_reinforcement_counts):
                p1_value = p1_reinforcement_counts[u]
            if u < len(p2_reinforcement_counts):
                p2_value = p2_reinforcement_counts[u]

            p1_reinforcements[u] = torch.full((self.HEIGHT, self.WIDTH), p1_value, dtype=torch.float32)
            p2_reinforcements[u] = torch.full((self.HEIGHT, self.WIDTH), p2_value, dtype=torch.float32)

            
            for s in range(self.N_UNIT_STATUSES):
                p_units[0].append(np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32))
                p_units[1].append(np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32))


        # Victory Points Channels
        for v in self.victory_p1:
            x = v[0]
            y = v[1]
            p1_victory[x][y] = 1

        for v in self.victory_p2:
            x = v[0]
            y = v[1]
            p2_victory[x][y] = 1


        # Unit Placement Channels
        for p in [0,1]: 
            # for each player check each unit status
            for unit in self.available_units[p]:
                x = unit.tile_x
                y = unit.tile_y

                u_type = unit.unit_type()
                status = 0
                type_i = (u_type-1)*self.N_UNIT_STATUSES

                p_units[p][type_i + status][x][y] = 1  
            
            for unit in self.moved_units[p]:
                x = unit.tile_x
                y = unit.tile_y

                u_type = unit.unit_type()
                status = 1
                type_i = (u_type-1)*self.N_UNIT_STATUSES
                
                p_units[p][type_i + status][x][y] = 1
            
            for unit in self.atacked_units[p]:
                x = unit.tile_x
                y = unit.tile_y

                u_type = unit.unit_type()
                status = 2
                type_i = (u_type-1)*self.N_UNIT_STATUSES
                
                p_units[p][type_i + status][x][y] = 1

        # Player Channel
        player_plane = np.ones((self.HEIGHT,self.WIDTH), dtype=np.int32)
        if self.current_player == 2:
            player_plane.fill(-1)

        player_plane = torch.unsqueeze(torch.as_tensor(player_plane,dtype=torch.float32), 0)

        # Phase plane
        phase = self.current_phase
        state_phase = torch.full((self.HEIGHT, self.WIDTH), phase, dtype=torch.float32)
        state_phase = torch.unsqueeze(state_phase, 0)

        # Turn plane
        turn = self.current_turn
        state_turn = torch.full((self.HEIGHT, self.WIDTH), turn, dtype=torch.float32)
        state_turn = torch.unsqueeze(state_turn, 0)

        # Final operations
        p1_victory = torch.unsqueeze(torch.as_tensor(p1_victory, dtype=torch.float32), 0)
        p2_victory = torch.unsqueeze(torch.as_tensor(p2_victory, dtype=torch.float32), 0)

        p1_reinforcements = torch.stack(p1_reinforcements, dim=0)
        p2_reinforcements = torch.stack(p2_reinforcements, dim=0)

        p1_units = torch.as_tensor(np.array(p_units[0]), dtype=torch.float32)
        p2_units = torch.as_tensor(np.array(p_units[1]), dtype=torch.float32)


        stack_list = []
        if self.use_terrain:
            terrain_list = [atack_modifiers, defense_modifiers, movement_costs]
            stack_list.extend(terrain_list)
    
        core_list = [p1_victory, p2_victory, p1_units, p2_units, p1_reinforcements, p2_reinforcements, state_phase, state_turn, player_plane]
        stack_list.extend(core_list)
        new_state = torch.concat(stack_list, dim=0)

        state_image = torch.unsqueeze(new_state, 0)# add batch size to the dimensions

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
    
##########################################################################
# -------------------------                   -------------------------- #
# ------------------------  UTILITY FUNCTIONS  ------------------------- #
# -------------------------                   -------------------------- #
##########################################################################

    def clone(self):
        return copy.deepcopy(self)
    
    def string_representation(self):
        
        string = "\n   "
        for k in range(self.WIDTH):
            string += (" " + format(k+1, '02') + " ")
        
        string += "\n  |"
        for k in range(self.WIDTH-1):
            string += "---|"

        string += "---|\n"

        for i in range(self.HEIGHT):
            string += format(i+1, '02') + "| "
            for j in range(self.WIDTH):
                mark = " "
                if self.board[i][j].victory == 1:
                    mark = colored("V", "cyan")
                elif self.board[i][j].victory == 2:
                    mark = colored("V", "yellow")

                unit = self.board[i][j].unit
                if unit:    
                    type = unit.unit_type()
                    if type==1:
                        if unit.player == 1:
                            mark=colored("S", "green")
                        else:
                            mark=colored("S", "red")
                    elif type==2:
                        if unit.player == 1:
                            mark=colored("T", "blue")
                        else:
                            mark=colored("T", "red")

                string += mark + ' | '
            string += "\n"

            if(i<self.HEIGHT-1):
                string += "  |"
                for k in range(self.WIDTH-1):
                    string += "---|"
                string += "---|\n"
            else:
                string += "   "
                for k in range(self.WIDTH-1):
                    string += "--- "
                string += "--- \n"

        string += "=="
        for k in range(self.WIDTH):
            string += "===="
        string += "==\n"

        return string

    def string_action(self, action_coords):

        parsed_action = self.parse_action(action_coords)

        act = parsed_action[0]
        unit_type = parsed_action[1]
        start = parsed_action[2]
        dest = parsed_action[3]

        string = ""
        if (act == 0): # placement
            string = "Unit placement phase: Placing a "
            if unit_type == 1: 
                string += "Soldier "
            elif unit_type == 2:
                string += "Tank "
            else:
                string += "Weird Unknown Unit "
            
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
    
##########################################################################
# ------------------------                     ------------------------- #
# -----------------------  USER PLAY FUNCTIONS  ------------------------ #
# ------------------------                     ------------------------- #
##########################################################################

    def user_p1_position_units(self):
        
        tmp_list = self.reinforcements[0].copy() #iterating over mutable object leads to bugs
        for unit in tmp_list:
            if unit == 1:
                unit_name = "Soldier"
            elif unit == 2:
                unit_name = "Tank"
            while(True):
                x = int(input("P1: You have a " + unit_name + " to place.\nPlease choose the row where you want to place it:"))
                while x<1 or x>self.HEIGHT:
                    x = int(input("You must choose a row inside the board:"))
                y = int(input("And the collumn:"))
                while y> math.floor(self.WIDTH/2) or y<0:
                    y = int(input("You must choose a collumn on your side of the board:"))
            
                if (self.board[x-1][y-1].unit):
                    print("There is a unit there already, try again.")
                    continue
                else:
                    break

            if unit==1:
                new_unit = Soldier(1,x-1,y-1)
                self.reinforcements[0].remove(1)
            elif unit==2:
                new_unit = Tank(1,x-1,y-1)
                self.reinforcements[0].remove(2)

            self.available_units[0].append(new_unit)
            self.board[x-1][y-1].place_unit(new_unit)
            self.print_board()
        
        print("P1 Positioning done.")
        print("")
        return

    def user_p2_position_units(self):
        
        tmp_list = self.reinforcements[1].copy() #iterating over mutable object leads to bugs

        for unit in tmp_list:
            if unit == 1:
                unit_name = "Soldier"
            elif unit == 2:
                unit_name = "Tank"
            while(True):
                x = int(input("P2: You have a " + unit_name + " to place.\nPlease choose the row where you want to place it:"))
                while x<1 or x>self.HEIGHT:
                    x = int(input("You must choose a row inside the board:"))
                y = int(input("And the collumn:"))
                while (y<= math.ceil(self.WIDTH/2) or y>self.WIDTH):
                    y = int(input("You must choose a collumn on your side of the board:"))

                if (self.board[x-1][y-1].unit):
                    print("There is a unit there already, try again.")
                    continue
                else:
                    break


            if unit==1:
                new_unit = Soldier(2,x-1,y-1)
                self.reinforcements[1].remove(1)
            elif unit==2:
                new_unit = Tank(2,x-1,y-1)
                self.reinforcements[1].remove(2)
                
            self.available_units[1].append(new_unit)
            self.board[x-1][y-1].place_unit(new_unit)
            self.print_board()

        print("P2 Positioning done.")
        print("")
        return

    def move_units(self, player):
        print("P" + str(player) + ": Movement phase.")
        

        tmp_list = self.available_units[player-1].copy() #iterating over mutable object leads to bugs
        for unit in tmp_list:
            pos_x = unit.tile_x
            pos_y = unit.tile_y
            mov = unit.mov_points
            invalid = 1
            while(invalid):
                x = int(input("P" + str(player) + ": You have a " + unit.unit_name() + " at (" + str(pos_x + 1) + "," + str(pos_y + 1) + ")"
                + " with " + str(mov) + " movement points.\nPlease choose the row you want to move it to:"))
                y = int(input("And the collumn:"))
                if ((abs(x-(pos_x+1)))+(abs(y-(pos_y+1)))<=mov and x>=1 and y>=1 and x<=self.HEIGHT and y <=self.WIDTH and
                    (((pos_x == (x-1)) and (pos_y == (y-1))) or self.board[x-1][y-1].unit is None)):
                    invalid=0
                else:
                    print("Invalid movement.Try again.")

            self.available_units[player-1].remove(unit)

            if (pos_x != (x-1)) or (pos_y != (y-1)):
                self.board[pos_x][pos_y].unit = None
                unit.move_to(x-1,y-1)
                self.board[x-1][y-1].unit = unit

            self.moved_units[player-1].append(unit)

            self.print_board()

        print("P" + str(player) + " Movement done.")
        print("")  
        return

    def fight(self, player):
        
        enemy_player = (not(player-1)) + 1

        print("P" + str(player) + ": Fight phase.")
        tmp_list = self.moved_units[player-1].copy() #iterating over mutable object leads to bugs
        for unit in tmp_list:
            atacked = False
            adjacent_units, _ = self.check_adjacent_units(unit, enemy_player)
            for enemy in adjacent_units:
                enemy_type = enemy.unit_type()
                self.print_board()
                
                answer = input("Your "  + unit.unit_name() + " at (" + str(unit.tile_x+1) + "," + str(unit.tile_y+1) +  ") " +  "has an enemy "
                + enemy.unit_name() + " nearby, at (" + str(enemy.tile_x+1) + "," + str(enemy.tile_y+1) + ").\nDo you want to fight it?(y/n)")
                if answer == "y":
                    self.resolve_combat(unit, enemy)
                    atacked = True
                    self.print_board()
                    break
                elif answer == "n":
                    continue
                else:
                    print("Bad answer.Exiting")
                    exit()
            if not atacked:
                self.moved_units[player-1].remove(unit)
                self.atacked_units[player-1].append(unit)
            
        print("P" + str(player) + " Combat done.")
        print("")
                
        return

    def play_user_vs_user(self):
        # WARNING: These funtions are not being maintained
        print("\n\nWARNING: This function is not being mantained and might be broken.\n")

        self.reset_env()

        self.print_board()

        self.user_p1_position_units()
        self.user_p2_position_units()

        while(self.current_turn<=self.TURNS):
            print("\nTurn: " + str(self.current_turn))
            
            self.move_units(1)
            self.fight(1)

            self.move_units(2)
            self.fight(2)

            print("End of turn")
            exit = input("Do you wish to continue playing?(y/n)\n")
            if exit == "n":
                break

            self.current_turn+=1
            self.new_turn()

        self.check_victory(show=True)

        return
    


