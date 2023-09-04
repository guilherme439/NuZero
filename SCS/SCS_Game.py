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

    PHASES = 2              # Check 'update_game_env()' 
    SUB_PHASES = 4
    STAGES = 10                

    N_PLAYERS = 2

    N_UNIT_STATUSES = 3     # Available, Moved, Attacked
    N_UNIT_STATS = 3        # Attack , Defense, Movement

    def __init__(self, game_config_path=""):
        
        # ------------------------------------------------------------ #
        # --------------------- INITIALIZATION  ---------------------- #
        # ------------------------------------------------------------ #
        self.turns = 0
        self.stacking_limit = 0

        self.rows = 0
        self.columns = 0
        self.board = []

        self.current_player = 1
        self.current_phase = 0
        self.current_sub_phase = 0
        self.current_stage = -2   
        self.current_turn = 0


        self.available_units = [[],[]]      # Units that have not move yet
        self.moved_units = [[],[]]          # Units that moved but didn't attack
        self.attacked_units = [[],[]]       # Units that already attacked

        self.target_tile = None
        self.attackers = []

        self.victory_p1 = []
        self.victory_p2 = []
        self.n_vp = [0, 0]

        self.all_reinforcements = {0:[], 1:[]}
        self.current_reinforcements = {0:[], 1:[]}

        self.p1_last_index = 0
        self.p2_first_index = 0

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

        # Reinforcements
        self.placement_planes = 1
        # Movement
        self.movement_planes = 6 * self.stacking_limit
        # Fighting
        self.choose_target_planes = 1
        self.choose_attackers_planes = self.stacking_limit
        self.confirm_attack_planes = 1
        self.fight_planes = self.choose_target_planes + self.choose_attackers_planes + self.confirm_attack_planes
        # Other
        self.no_move_planes = self.stacking_limit
        self.no_fight_planes = self.stacking_limit

        self.total_action_planes = \
        self.placement_planes + \
        self.movement_planes + \
        self.fight_planes + \
        self.no_move_planes + \
        self.no_fight_planes
        
        self.action_space_shape = (self.total_action_planes , self.rows , self.columns)
        self.num_actions     =     self.total_action_planes * self.rows * self.columns


        ## STATE REPRESENTATION
        
        # Terrain
        self.n_terrain_channels = 3 # atack, defense, movement
        # Victory points
        self.n_vp_channels = self.N_PLAYERS
        # Units
        self.n_unit_representation_channels = self.N_UNIT_STATS * self.stacking_limit * self.N_UNIT_STATUSES * self.N_PLAYERS
        # Reinforcements
        self.n_reinforcements = 3  # Number of reinforcements that are represented per player
        self.n_reinforcement_channels = self.n_reinforcements * self.N_UNIT_STATS
        # For each unit there will be 2 sets of planes:
        # the first representing the unit itself and
        # the second representing how many turns until that unit arrives
        self.n_duration_channels = self.n_reinforcement_channels
        self.n_total_reinforcement_channels = (self.n_reinforcement_channels + self.n_duration_channels) * self.N_PLAYERS

        # Attack
        self.n_target_tile_channels = 1
        self.n_attacker_channels = self.stacking_limit
        self.n_attack_channels = self.n_target_tile_channels + self.n_attacker_channels
        # Features
        self.n_sub_phase_channels = self.SUB_PHASES
        self.n_turn_channels = 1
        self.n_player_channels = 1
        self.n_feature_channels = self.n_sub_phase_channels + self.n_turn_channels + self.n_player_channels

        self.total_dims = \
        self.n_vp_channels + \
        self.n_unit_representation_channels + \
        self.n_total_reinforcement_channels + \
        self.n_feature_channels + \
        self.n_terrain_channels + \
        self.n_attack_channels

        self.game_state_shape = (self.total_dims, self.rows, self.columns)

        return
    
##########################################################################
# -------------------------                    ------------------------- #
# -----------------------  GET AND SET METHODS  ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def get_name(self):
        return "SCS"

    def get_board(self):
        return self.board
    
    def getBoardColumns(self):
        return self.columns

    def getBoardRows(self):
        return self.rows    

    def get_current_player(self):
        return self.current_player
    
    def get_terminal_value(self):
        return self.terminal_value

    def get_length(self):
        return self.length

    def get_action_space_shape(self):
        return self.action_space_shape

    def get_num_actions(self):
        return self.num_actions

    def state_shape(self):
        return self.game_state_shape

    def get_state_from_history(self, i):
        return self.state_history[i]

    def store_state(self, state):
        self.state_history.append(state)
        return

    def store_player(self, player):
        self.player_history.append(player)
        return
    
    def store_action(self, action_coords):
        self.action_history.append(action_coords)
          
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
        
        # PLANE DEFINITIONS
        placement_planes = np.zeros((self.placement_planes, self.rows, self.columns), dtype=np.int32)

        movement_planes = np.zeros((self.movement_planes, self.rows, self.columns), dtype=np.int32)

        choose_target_planes = np.zeros((self.choose_target_planes, self.rows, self.columns), dtype=np.int32)
        choose_attackers_planes = np.zeros((self.choose_attackers_planes, self.rows, self.columns), dtype=np.int32)
        confirm_attack_planes = np.zeros((self.confirm_attack_planes, self.rows, self.columns), dtype=np.int32)

        no_move_planes = np.zeros((self.no_move_planes, self.rows, self.columns), dtype=np.int32)
        no_fight_planes = np.zeros((self.no_fight_planes, self.rows, self.columns), dtype=np.int32)
        
        
        # PLACING REINFORCEMENTS
        if self.current_sub_phase == 0:
            # In this sub_phase the player places his reinforcements

            next_reinforcement = self.current_reinforcements[self.current_player-1][self.current_turn][0]
            arraival_locations = next_reinforcement.get_arraival_locations()
            for (row, col) in arraival_locations:
                tile = self.board[row][col]
                # can not place on tiles controlled by the other player or that are already full.
                if not ( (tile.player == self.opponent(player)) or (tile.stacking_number() == self.stacking_limit) ):
                    placement_planes[0][row][col] = 1       
        
        # MOVING
        elif self.current_sub_phase == 1:
            # In this sub_phase the player can either choose a unit to move or end it's movement

            for unit in self.available_units[player-1]:
                (x, y) = unit.position
                tile = self.board[x][y]

                s = tile.get_stacking_level(unit)
                no_move_planes[s][x][y] = 1 # no move action

                tiles = self.check_tiles((x,y))
                movements = self.check_mobility(unit, consider_other_units=True)

                for i in range(len(tiles)):
                    tile = tiles[i]
                    if (tile):
                        if(movements[i]):
                            plane_index = i * self.stacking_limit + s
                            movement_planes[plane_index][x][y] = 1

        # ATTACKING
        elif self.current_sub_phase == 2:
            # In this sub_phase the player can either choose a target or select a unit as done attacking
            
            for unit in self.moved_units[player-1]:
                (x, y) = unit.position
                tile = self.board[x][y]

                s = tile.get_stacking_level(unit)
                no_fight_planes[s][x][y] = 1 # no fight action

                enemy_player = self.opponent(player)
                enemy_units = self.check_adjacent_units(unit.position, enemy_player)
                for enemy_unit in enemy_units:
                    (row, col) = enemy_unit.position
                    choose_target_planes[0][row][col] = 1 # select target action

        elif self.current_sub_phase == 3:
            # In this sub_phase the player can either select a unit to atack with or confirm the attack

            selectable_units = self.check_adjacent_units(self.target_tile.position, player)
            for unit in selectable_units.copy():
                if (unit in self.attackers) or (unit in self.attacked_units[player-1]):
                    selectable_units.remove(unit)

            for unit in selectable_units:
                (x, y) = unit.position
                tile = self.board[x][y]
                s = tile.get_stacking_level(unit)
                choose_attackers_planes[s][x][y] = 1 # choose attacker action

            num_attackers = len(self.attackers)
            if num_attackers > 0:
                (x,y) = self.target_tile.position
                confirm_attack_planes[0][x][y] = 1 # confirm attack action
        
        else:
            print("Error in possible_actions! Exiting")
            exit()

        planes_list = [placement_planes, movement_planes, choose_target_planes, choose_attackers_planes, confirm_attack_planes, no_move_planes, no_fight_planes]
        valid_actions_mask = np.concatenate(planes_list)
        return valid_actions_mask
                      
    def parse_action(self, action_coords):
        act = None           # Represents the type of action
        start = (None, None) # Starting point of the action
        stacking_lvl = None  # Stacking level
        dest = (None, None)  # Destination point for the action

        current_plane = action_coords[0]

        # Plane borders
        placement_limit = self.placement_planes
        movement_limit = placement_limit + self.movement_planes
        target_limit = movement_limit + self.choose_target_planes
        attackers_limit = target_limit + self.choose_attackers_planes
        confirm_limit = attackers_limit + self.confirm_attack_planes
        no_move_limit = confirm_limit + self.no_move_planes
        no_fight_limit = no_move_limit + self.no_fight_planes


        # PLACEMENT PLANES
        if current_plane < placement_limit:
            act = 0
            start = (action_coords[1], action_coords[2])

        # MOVEMENT PLANES
        elif current_plane < movement_limit:
            act = 1
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

            plane_index = current_plane - placement_limit
            stacking_lvl = plane_index % self.stacking_limit
            direction = plane_index // self.stacking_limit

            # n, ne, se, s, sw, nw
            if direction == 0:    # N
                dest = self.get_n_coords(start)

            elif direction == 1:  # NE
                dest = self.get_ne_coords(start)

            elif direction == 2:  # SE
                dest = self.get_se_coords(start)

            elif direction == 3:  # S
                dest = self.get_s_coords(start)

            elif direction == 4:  # SW
                dest = self.get_sw_coords(start)

            elif direction == 5:  # NW
                dest = self.get_nw_coords(start)
            else:
                print("Problem parsing action...Exiting")
                exit()
            
        # FIGHT PLANES
        elif current_plane < target_limit:
            act = 2
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)

        elif current_plane < attackers_limit:
            act = 3
            start = (action_coords[1], action_coords[2])

            plane_index = current_plane - target_limit
            stacking_lvl = plane_index

        elif current_plane < confirm_limit:
            act = 4
            start = (action_coords[1],action_coords[2])

        # NO_MOVE PLANE
        elif current_plane < no_move_limit:
            act = 5
            plane_index = current_plane - confirm_limit
            stacking_lvl = plane_index
            start = (action_coords[1],action_coords[2])

        # NO_FIGHT PLANE
        elif current_plane < no_fight_limit:
            act = 6
            plane_index = current_plane - no_move_limit
            stacking_lvl = plane_index
            start = (action_coords[1],action_coords[2])

        else:
            print("Problem parsing action...Exiting")
            exit()

        return (act, start, stacking_lvl, dest)

    def play_action(self, action_coords):
        (act, start, stacking_lvl, dest) = self.parse_action(action_coords)

        if (act == 0): # Placement
            player = self.current_player
            turn = self.current_turn
            new_unit = self.current_reinforcements[player-1][turn].pop(0)
            new_unit.move_to(start, 0)
            self.available_units[player-1].append(new_unit)

            tile = self.board[start[0]][start[1]]
            tile.place_unit(new_unit)

        elif (act == 1): # Movement
            start_tile = self.board[start[0]][start[1]]
            unit = start_tile.get_unit_by_level(stacking_lvl)

            if start != dest:
                dest_tile = self.board[dest[0]][dest[1]]
                
                terrain = dest_tile.get_terrain()
                cost = terrain.cost

                unit.move_to(dest, cost)
                dest_tile.place_unit(unit)
                start_tile.remove_unit(unit)

                # Ends the movement of units who don't have enough movement points
                # to reduce total number of decisions that need to be taken per game
                if not any(self.check_mobility(unit, consider_other_units=False)): 
                    self.end_movement(unit)
                # This verification is done here to avoid checking all the units, or keeping an 'updated_units_list'
           
            else:
                print("Problem playing action.\n \
                      Probably there is a bug in possible_actions().\n \
                      Exiting")
                exit()

        elif (act == 2): # Choosing target
            target_tile = self.board[start[0]][start[1]]
            self.target_tile = target_tile
        
        elif (act == 3): # Choosing attacker
            tile = self.board[start[0]][start[1]]
            unit = tile.get_unit_by_level(stacking_lvl)
            self.attackers.append(unit)

        elif (act == 4): # Confirming attack
            self.resolve_combat()
            self.target_tile = None # clear target tile
            self.attackers.clear()  # reset attackers

        elif (act == 5): # No movement
            tile = self.board[start[0]][start[1]]
            unit = tile.get_unit_by_level(stacking_lvl)
            self.end_movement(unit)

        elif (act == 6): # No fighting
            tile = self.board[start[0]][start[1]]
            unit = tile.get_unit_by_level(stacking_lvl)
            self.end_fighting(unit)

        else:
            print("Unknown action. Exiting...")
            exit()

        return

    # --------------- ENVIRONMENT --------------- #

    def reset_env(self):
        self.current_player = 1  
        self.current_phase = 0   
        self.current_sub_phase = 0
        self.current_stage = -2 
        self.current_turn = 0

        self.target_tile = None
        self.attackers = []

        self.length = 0
        self.terminal_value = 0
        self.terminal = False

        for p in [0,1]:
            self.available_units[p].clear()
            self.moved_units[p].clear()
            self.attacked_units[p].clear()

        self.current_reinforcements = deepcopy(self.all_reinforcements)
        
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
        # Two players: P1 and P2
        # Each player's turn has two phases: Movement, Fighting
        # Each phase has 2 sub-phases:
        # Movement sub-phases -> reinforcement | movement
        # Fighting sub-phases -> choosing target | choosing attackers
        # This results in 4 total sub-phases for each player.

        # Turn 0 happens before game start, and is where both player place their initial troops.
        # Turn 0 only has 1 sub-phase for each player (reinforcement)

        # I call "stage" to each unique sub-phase of the game.
        # There are a total of 10 stages:
        # 2 stages in turn 0 (1 for each player's reinforcement sub-phase)
        # 8 stages in other turns (4 for each player)

        done = False
        previous_player = self.current_player
        previous_stage = self.current_stage
        stage = previous_stage

        # Turn 0 stages are represented with -2 and -1
        while True:
            match stage:
                case -2:
                    if self.current_turn != 0:
                        print("Sanity check went wrong. Something wrong with game environment.")
                        exit()
                    
                    if self.player_ended_reinforcements(1, self.current_turn):
                        stage+=1
                        continue

                case -1:
                    if self.current_turn != 0:
                        print("Sanity check went wrong. Something wrong with game environment.")
                        exit()
                    
                    if self.player_ended_reinforcements(2, self.current_turn):
                        self.current_turn+=1
                        stage+=1
                        continue
                
                case 0: # P1 reinforcements
                    if self.player_ended_reinforcements(1, self.current_turn):
                        stage+=1
                        continue
                
                case 1: # P1 movement
                    if self.player_ended_movement(1):
                        stage+=1
                        continue
                    
                case 2: # P1 choosing target
                    if self.player_done_attacking(1):
                        stage = 4
                        continue
                        
                    elif self.player_chose_target(1):
                        stage+=1
                        continue

                case 3: # P1 selecting attackers
                    if self.player_confirmed_attack(1):
                        stage = 2
                        continue
                    
                case 4: # P2 reinforcements
                    if self.player_ended_reinforcements(2, self.current_turn):
                        stage+=1
                        continue
                    
                case 5: # P2 movement
                    if self.player_ended_movement(2):
                        stage+=1
                        continue                                                      

                case 6: # P2 choosing target
                    if self.player_done_attacking(2):
                        if self.current_turn+1 > self.turns:
                            done = True
                            break
                        self.current_turn+=1
                        stage = 0
                        self.new_turn()
                        continue
            
                    elif self.player_chose_target(2):
                        stage+=1
                        continue         
                    
                case 7: # P2 selecting attackers
                    if self.player_confirmed_attack(2):
                        stage = 6
                        continue
            break
            
            
        if(done):
            self.terminal = True
            self.terminal_value = self.check_terminal()
    
        # ------------------------------------    

        p1_stages = (-2,0,1,2,3)
        p2_stages = (-1,4,5,6,7)

        if stage in p1_stages:
            self.current_player = 1
        elif stage in p2_stages:
            self.current_player = 2
        else:
            print("Error in function: \'update_game_env()\'.Exiting")
            exit()


        reinforcement_stages = (-2,-1,0,4)
        movement_stages = (1,5)
        choosing_target_stages = (2,6)
        choosing_attackers_stages = (3,7)

        if stage in reinforcement_stages:
            self.current_sub_phase = 0
        elif stage in movement_stages:
            self.current_sub_phase = 1
        elif stage in choosing_target_stages:
            self.current_sub_phase = 2
        elif stage in choosing_attackers_stages:
            self.current_sub_phase = 3
        else:
            print("Error in function: \'update_game_env()\'.Exiting")
            exit()

        
        if self.current_sub_phase in (0,1):
            self.current_phase = 0
        elif self.current_sub_phase in (2,3):
            self.current_phase = 1
        else:
            print("Error in function: \'update_game_env()\'.Exiting")
            exit()

        self.current_stage = stage

        return done

    def new_turn(self):
        # Resets units status before new turn
        self.available_units = self.attacked_units.copy()
        self.attacked_units = [[], []]

        for p in [0,1]:
            for unit in self.available_units[p]:
                unit.reset_mov()
                unit.set_status(0)

        return  

    def check_terminal(self):
        p1_captured_points = 0
        p2_captured_points = 0
        for point in self.victory_p1:
            vic_p1 = self.board[point[0]][point[1]]
            if vic_p1.player == 2:
                p2_captured_points +=1
        for point in self.victory_p2:
            vic_p2 = self.board[point[0]][point[1]]
            if vic_p2.player == 1:
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

    def player_ended_reinforcements(self, player, turn):
        player_index = player-1
        turn_index = turn
        return self.current_reinforcements[player_index][turn_index] == []
    
    def player_ended_movement(self, player):
        player_index = player-1
        return self.available_units[player_index] == []
        
    def player_done_attacking(self, player):
        player_index = player-1
        return self.moved_units[player_index] == []        

    def player_chose_target(self, player):
        return (self.target_tile is not None)
    
    def player_confirmed_attack(self, player):
        return (self.target_tile is None)      

    def end_movement(self, unit):
        # End movement
        unit.set_status(1)
        self.moved_units[unit.player-1].append(unit)
        self.available_units[unit.player-1].remove(unit)

        # Since enemies can not move during my turn we can
        # mark isolated units as done fighting to reduce
        # the total number of decisions that need to be taken
        enemy = self.opponent(unit.player)
        enemy_units = self.check_adjacent_units(unit.position, enemy)
        if len(enemy_units) == 0:
            self.end_fighting(unit)

    def end_fighting(self, unit):
        unit.set_status(2)
        self.attacked_units[unit.player-1].append(unit)
        self.moved_units[unit.player-1].remove(unit)

    # ------------------ COMBAT ------------------ #
    
    def destroy_unit(self, unit):
        (x, y) = unit.position
        self.board[x][y].remove_unit(unit)
        player_index = unit.player-1
        
        # Global reference list to each status caused problems, so I use a local list instead
        all_statuses = [self.available_units, self.moved_units, self.attacked_units]

        try:
            all_statuses[unit.status][player_index].remove(unit)
        except ValueError:
            print("Can not destroy this unit.")
            print("Possible error tracking unit's status. Exiting")
            exit()

        return

    def resolve_combat(self):
        attacking_player = self.current_player
        defending_player = self.opponent(attacking_player)
        
        # DEFENSE
        target_tile = self.target_tile
        defense_modifier = target_tile.get_terrain().defense_modifier
        total_defense = 0
        defending_units = target_tile.units
        for unit in defending_units:
            total_defense += unit.defense
        total_defense = total_defense * defense_modifier

        # ATTACK
        total_attack = 0
        for unit in self.attackers:
            unit_tile = self.board[unit.position[0]][unit.position[1]]
            attack_modifier = unit_tile.get_terrain().attack_modifier
            total_attack += unit.attack * attack_modifier
            self.end_fighting(unit)

        # RESULTS
        attacker_losses, defender_losses = self.get_combat_results(total_attack, total_defense)

        for loss in range(attacker_losses):
            unit = self.get_strongest_attacker(self.attackers)
            self.destroy_unit(unit)

        for loss in range(defender_losses):
            unit = self.get_strongest_defender(defending_units)
            self.destroy_unit(unit)

    def get_combat_results(self, total_attack, total_defense):
        # In the future this function should use a combat table
        attacker_losses = 0
        defender_losses = 0

        if total_attack > total_defense:    # defender loses a unit
            defender_losses = 1

        elif total_attack < total_defense:  # attacker loses a unit
            attacker_losses = 1
        
        else:                               # both lose a unit
            attacker_losses = 1
            defender_losses = 1

        return attacker_losses, defender_losses

    # ------------------ OTHER ------------------ #

    def check_tiles(self, coords):
        ''' Clock-wise rotation order '''

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

        tiles = self.check_tiles(unit.position)
        can_move = [False for i in range(len(tiles))]

        for i in range(len(tiles)):
            tile = tiles[i]
            if tile:
                cost = tile.get_terrain().cost
                dif = unit.mov_points - cost
                if dif >= 0:
                    can_move[i] = True
                    if consider_other_units and ((tile.stacking_number() == self.stacking_limit) or (tile.player == self.opponent(unit.player))):
                        can_move[i] = False

        return can_move
    
    def check_adjacent_units(self, position, player):

        tiles = self.check_tiles(position)
        adjacent_units = []
        for i in range(len(tiles)):
            tile = tiles[i]
            if tile:
                for unit in tile.units:
                    if (unit.player==player):
                        adjacent_units.append(unit)

        return adjacent_units

    
##########################################################################
# -------------------------                   -------------------------- #
# ------------------------   UTILITY METHODS   ------------------------- #
# -------------------------                   -------------------------- #
##########################################################################

    def is_terminal(self):
        return self.terminal

    def opponent(self, player):
        ''' 1 -> 2    |    2 -> 1 '''
        return (not(player-1)) + 1 

    def define_board_sides(self):

        # Calculate the indexes that define each side of the board
        if self.columns % 2 != 0:
            middle_index = math.floor(self.columns/2)
            self.p1_last_index = middle_index-1
            self.p2_first_index = middle_index+1
        else:
            # if number of columns is even there are two middle columns: one on the right and one on the left
            mid = int(self.columns/2)
            left_side_collumn = mid
            right_side_collumn = mid + 1
            left_index = left_side_collumn - 1
            right_index = right_side_collumn - 1
            
            # For boards with even columns we separate the center by one more column
            self.p1_last_index = max(0, left_index-1)
            self.p2_first_index = min(self.columns-1, right_index+1)

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

    def get_strongest_defender(self, units_list):
        # returns the unit from the list which has the highest defense
        # In case of a draw, attack and movement allowance are used to select the unit

        strongest_unit = units_list[0]
        for unit in units_list:
            if unit.defense > strongest_unit.defense:
                strongest_unit = unit
            elif unit.defense == strongest_unit.defense:
                if unit.attack > strongest_unit.attack:
                    strongest_unit = unit
                elif unit.attack == strongest_unit.attack:
                    if unit.mov_allowance > strongest_unit.mov_allowance:
                        strongest_unit = unit

        return strongest_unit
    
    def get_strongest_attacker(self, units_list):
        # returns the unit from the list which has the highest attack
        # In case of a draw, defense and movement allowance are used to select the unit

        strongest_unit = units_list[0]
        for unit in units_list:
            if unit.attack > strongest_unit.attack:
                strongest_unit = unit
            elif unit.attack == strongest_unit.attack:
                if unit.defense > strongest_unit.defense:
                    strongest_unit = unit
                elif unit.defense == strongest_unit.defense:
                    if unit.mov_allowance > strongest_unit.mov_allowance:
                        strongest_unit = unit

        return strongest_unit
    
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
            represented_units = 0
            for turn in range(len(reinforcements)):
                turns_left = turn - self.current_turn
                relative_importance = (self.turns + 1) - turns_left
                normalized_importance = relative_importance / (self.turns + 1)

                turn_reinforcements = reinforcements[turn]
                for unit in turn_reinforcements:
                    arraival_locations = unit.get_arraival_locations()
                    attack_plane = torch.zeros((1, self.rows, self.columns))
                    defense_plane = torch.zeros((1, self.rows, self.columns))
                    movement_plane = torch.zeros((1, self.rows, self.columns))
                    for (row, col) in arraival_locations:
                        attack_plane[0][row][col] = unit.attack
                        defense_plane[0][row][col] = unit.defense
                        movement_plane[0][row][col] = unit.mov_points
                    
                    stats_planes = torch.cat((attack_plane, defense_plane, movement_plane))
                    duration_planes = torch.full((self.N_UNIT_STATS, self.rows, self.columns), normalized_importance)
                    unit_planes = torch.cat((stats_planes, duration_planes))

                    if player_reinforcements[player] is None:
                        player_reinforcements[player] = unit_planes
                    else:
                        player_reinforcements[player] = torch.cat((player_reinforcements[player], unit_planes))

                    represented_units +=1
                    if represented_units == self.n_reinforcements:
                        break
                if represented_units == self.n_reinforcements:
                        break
                
            # If the loop ends without reaching a "break" it means we need to fill the rest with "empty" units 
            else: # This else belongs to the "for" loop not the "if" statement
                units_remaining = self.n_reinforcements - represented_units
                for empty_unit in range(units_remaining):
                    unit_planes = torch.zeros((self.N_UNIT_STATS * 2, self.rows, self.columns))
                    if player_reinforcements[player] is None:
                        player_reinforcements[player] = unit_planes
                    else:
                        player_reinforcements[player] = torch.cat((player_reinforcements[player], unit_planes))


        p1_reinforcements = player_reinforcements[0]
        p2_reinforcements = player_reinforcements[1]

    
        # Victory Points Channels #
        p1_victory = torch.zeros((self.rows, self.columns))
        p2_victory = torch.zeros((self.rows, self.columns))

        for v in self.victory_p1:
            x = v[0]
            y = v[1]
            p1_victory[x][y] = 1

        for v in self.victory_p2:
            x = v[0]
            y = v[1]
            p2_victory[x][y] = 1

        p1_victory = torch.unsqueeze(p1_victory, 0)
        p2_victory = torch.unsqueeze(p2_victory, 0)


        # Unit Representation Channels #
        p1_units = torch.zeros(self.N_UNIT_STATS * self.stacking_limit * self.N_UNIT_STATUSES, self.rows, self.columns)
        p2_units = torch.zeros(self.N_UNIT_STATS * self.stacking_limit * self.N_UNIT_STATUSES, self.rows, self.columns)
        p_units = [p1_units, p2_units]
        for p in [0,1]: 
            # for each player check each unit status
            statuses_list = [self.available_units[p], self.moved_units[p], self.attacked_units[p]]
            for status_index in range(len(statuses_list)):
                unit_list = statuses_list[status_index]
                # within each status we represent each unit using their stacking level and stats
                for unit in unit_list:
                    (row, col) = unit.position
                    tile = self.board[row][col]
                    s = tile.get_stacking_level(unit)
                    stacking_offset = s * self.N_UNIT_STATS
                    status_offset = status_index * self.stacking_limit * self.N_UNIT_STATS
                    p_units[p][status_offset + stacking_offset + 0][row][col] = unit.attack
                    p_units[p][status_offset + stacking_offset + 1][row][col] = unit.defense
                    p_units[p][status_offset + stacking_offset + 2][row][col] = unit.mov_points
        
        # Target tile channel #
        target_tile_plane = torch.zeros((1, self.rows, self.columns))
        if self.target_tile is not None:
            (x, y) = self.target_tile.position
            target_tile_plane[0][x][y] = 1

        # Attackers channels #
        attackers = torch.zeros((self.stacking_limit, self.rows, self.columns))
        for unit in self.attackers:
            (x, y) = unit.position
            tile = self.board[x][y]
            stacking_lvl = tile.get_stacking_level(unit)
            attackers[stacking_lvl][x][y] = 1

        # Sub-Phase Channel #
        sub_phase_index = self.current_sub_phase
        sub_phase_planes = torch.zeros((self.SUB_PHASES, self.rows, self.columns))
        active_sub_phase = torch.ones((self.rows, self.columns))
        sub_phase_planes[sub_phase_index] = active_sub_phase
        
        # Turn Channel #
        turn_percent = self.current_turn/self.turns
        turn_plane = torch.full((self.rows, self.columns), turn_percent, dtype=torch.float32)
        turn_plane = torch.unsqueeze(turn_plane, 0)

        # Player Channel #
        player_plane = np.ones((self.rows,self.columns), dtype=np.int32)
        if self.current_player == 2:
            player_plane.fill(-1)

        player_plane = torch.unsqueeze(torch.as_tensor(player_plane,dtype=torch.float32), 0)

        # Final operations #
        stack_list = []

        terrain_list = [atack_modifiers, defense_modifiers, movement_costs]
        stack_list.extend(terrain_list)
    
        core_list = [p1_victory, p2_victory, p1_reinforcements, p2_reinforcements, p1_units, p2_units,
                     target_tile_plane, attackers, sub_phase_planes, turn_plane, player_plane]
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
        print("\n")
        print("---------" + ("----" * self.columns))
        print("\n")

        state = state_image[0]

        section_names = ["TERRAIN", "VICTORY POINTS", "P1_REINFORCEMENTS", "P2_REINFORCEMENTS",
                         "P1_UNITS", "P2_UNITS", "TARGET TILE", "ATTACKERS", "SUBPHASES", "TURN", "PLAYER"]
        section_sizes = [self.n_terrain_channels, self.n_vp_channels, self.n_total_reinforcement_channels//2, self.n_total_reinforcement_channels//2, 
                         self.n_unit_representation_channels//2, self.n_unit_representation_channels//2, self.n_target_tile_channels,
                         self.n_attacker_channels, self.n_sub_phase_channels, 1, 1]

        limit = 0
        section_index = 0
        for channel_idx in range(len(state)):
            channel = state[channel_idx]
            
            if channel_idx == limit:
                print("\n" + section_names[section_index])
                size = section_sizes[section_index]
                if torch.count_nonzero(state[limit:limit+size]) == 0:
                    print("\n(empty)\n")
                    empty_section = True
                else:
                    empty_section = False

                limit += size
                section_index += 1

            if not empty_section:
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
        
        for section_name, values in data_loaded.items():
            match section_name:
                case "Board_dimensions":
                    self.rows = values["rows"]
                    self.columns = values["columns"]
                    self.define_board_sides()

                case "Turns":
                    self.turns = values

                case "Stacking_limit":
                    self.stacking_limit = values

                case "Units":
                    for unit_name, properties in values.items():
                        id = properties["id"]
                        properties.pop("id")
                        units_by_id[id] = {}
                        units_by_id[id]["name"] = unit_name
                        units_by_id[id].update(properties)
            
                case "Reinforcements":
                    # Currently reinforcements can be placed on each player's side of the board
                    player_arraival_locations = [[], []]
                    for i in range(self.rows):
                        for j in range(self.columns):
                            location = (i,j)
                            if j <= self.p1_last_index:
                                player_arraival_locations[0].append(location)
                            elif j >= self.p2_first_index:
                                player_arraival_locations[1].append(location)


                    for p, reinforcements in values.items():
                        if len(reinforcements) != (self.turns + 1):
                            print("\nError in config.\n \
                                  Reinforcement schedule should have \'turns + 1\' entries.\n \
                                  In order to account for initial troop placement (turn 0).")
                            exit()
                        player_index = int(p[-1]) - 1
                        self.all_reinforcements[player_index] = []
                        self.current_reinforcements[player_index] = []
                        for turn_idx in range(len(reinforcements)):
                            turn_units = reinforcements[turn_idx]

                            self.all_reinforcements[player_index].append([])
                            self.current_reinforcements[player_index].append([])
                            for id in turn_units:
                                player = player_index + 1
                                name = units_by_id[id]["name"]
                                attack = units_by_id[id]["attack"]
                                defense = units_by_id[id]["defense"]
                                mov_allowance = units_by_id[id]["movement"]
                                image_path = units_by_id[id]["image_path"]
                                # In the future, configs should allow definition of custom arraival locations
                                unit_arraival_locations = player_arraival_locations[player_index]
                                new_unit = Unit(name, attack, defense, mov_allowance, player, unit_arraival_locations, image_path)
                                self.current_reinforcements[player_index][turn_idx].append(new_unit)

                    self.all_reinforcements = deepcopy(self.current_reinforcements)

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
                                self.board[i].append(Tile((i,j), terrain))

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
                                    self.board[i].append(Tile((i,j), terrain))
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
                        for player in range(len(loaded_vps)):
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
        string = ""
        # Horizontal line
        string += "\n ====="
        for k in range(self.columns):
            string += "==="
        string += "==\n"

        # Collumn numbers and top
        first_line_numbers = "\n     "
        top = "\n     "
        for k in range(self.columns):
            first_line_numbers += (format(k+1, '02') + " ")
            odd_col = k%2
            if odd_col:
                top += "   "
            else:
                top += "__ "

        string += first_line_numbers
        string += (top + "\n")

        for i in range(self.rows):
            first_line = "    "
            second_line = format(i+1, '02') + "  "
            for j in range(self.columns):
                tile = self.board[i][j]
                mark = "  "
                mark_text = "  "
                mark_color = "white"
                attributes=[]

                if tile.victory == 1:
                    mark_color = "cyan"
                    mark_text = " *"
                elif tile.victory == 2:
                    mark_color = "yellow"
                    mark_text = " *"

                s = tile.stacking_number()
                if s > 0:
                    number = str(s)
                    if s>9:
                        number = "X"
                    mark_text = "U" + number
                    
                    if tile.player == 1:
                        if mark_color == "white":
                            mark_color = "blue"
                        elif mark_color == "yellow":
                            mark_color = "green"
                    else:
                        if mark_color == "white":
                            mark_color = "red"
                        if mark_color == "cyan":
                            mark_color = "magenta"
                            attributes=["dark"]

                mark = colored(mark_text, mark_color, attrs=attributes)

                odd_col = j%2
                if odd_col:
                    first_line += '__'
                    second_line += mark
                else:
                    first_line += "/" + mark + "\\"
                    second_line += "\__/"

            string += (first_line + "\n")
            string += (second_line + "\n")

        # Bottom
        bottom = "     "
        for k in range(self.columns):
            odd_col = k%2
            if odd_col:
                bottom += "\__/"
            else:
                bottom += "  "

        string += (bottom + "\n")

        # Horizontal line
        string += "\n ====="
        for k in range(self.columns):
            string += "==="
        string += "==\n"

        return string

    def string_action(self, action_coords):

        parsed_action = self.parse_action(action_coords)

        act = parsed_action[0]
        start = parsed_action[1]
        stacking_lvl = parsed_action[2]
        dest = parsed_action[3]

        string = ""
        if (act == 0): # placement
            string = "Movement phase: Placing reinforcement "
        
            string = string + "at (" + str(start[0]+1) + "," + str(start[1]+1) + ")"

        elif (act == 1): # movement
            string = "Movement phase: Moving from (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "to (" + str(dest[0]+1) + "," + str(dest[1]+1) + ")"
        
        elif (act == 2): # choose target
            string = "Fighting phase: Targeting the tile at (" + str(start[0]+1) + "," + str(start[1]+1) + ")"

        elif (act == 3): # choose attacker
            string = "Fighting phase: Chose the unit in tile (" + str(start[0]+1) + "," + str(start[1]+1) + ") at stacking level:" + str(stacking_lvl) + ", to join the attack"

        elif (act == 4): # confirm attack
            string = "Fighting phase: Attack to tile (" + str(start[0]+1) + "," + str(start[1]+1) + ") confirmed"

        elif (act == 5): # no move
            string = "Movement phase: Unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "chose not to move"

        elif (act == 6): # no fight
            string = "Fighting phase: Unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "chose not to fight"

        else:
            string = "Unknown action..."

        #print(string)
        return string

##########################################################################
# -------------------------                    ------------------------- #
# ------------------------    LEGACY METHODS    ------------------------ #
# -------------------------                    ------------------------- #
##########################################################################

    def string_squared_representation(self):
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

                for unit in self.board[i][j].units:    
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