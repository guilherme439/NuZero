import copy
import math
import numpy as np
import torch
import enum
import sys
import gc
import time

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

    N_VP = 1 # Number of vicotry points
    N_UNIT_TYPES = 2
    N_UNIT_STATUSES = 3



    def __init__(self, r1=[1,1], r2=[1,1], use_terrain=True):


        self.board = []
        self.current_player = 1  # Two players: 1 and 2
        self.current_phase = 0   # Three phases: Placement, movement and fighting
        self.current_stage = 0   # Six stages: P1_Placement, P2_Placement, P1_Movement, P1_Fighting, P2_Movement, P2_Fighting
        self.current_turn = 1
        self.reinforcements = [[],[]]
        self.available_units = [[],[]]
        self.moved_units = [[],[]]
        self.atacked_units = [[],[]]

        self.use_terrain = use_terrain
        

        self.victory_p1 = [ [-1,-1] for _ in range(self.N_VP) ]
        self.victory_p2 = [ [-1,-1] for _ in range(self.N_VP) ]

        self.reinforcements_by_type = [[],[]]
        self.reinforcements_by_type[0] = r1 # number of soldiers , number of tanks... etc
        self.reinforcements_by_type[1] = r2

        self.reinforcements_as_list = [[],[]]
        self.reinforcements_as_list[0] = np.concatenate([np.repeat(i+1, self.reinforcements_by_type[0][i]) for i in range(len(self.reinforcements_by_type[0]))]).tolist()
        self.reinforcements_as_list[1] = np.concatenate([np.repeat(i+1, self.reinforcements_by_type[1][i]) for i in range(len(self.reinforcements_by_type[1]))]).tolist()         

        self.terrain_types = []
        
        # 3D Action Space Representation Info
        self.n_placement_planes = self.N_UNIT_TYPES 
        self.n_movement_planes = self.HEIGHT * self.WIDTH # each plane x corresponds to the action of moving to tile x (there are "size" tiles)
        self.n_fight_planes = 4 # {N, S, E, 0} directions
        self.n_extra_planes = 1 # No_Fight plane
        self.total_action_planes = self.n_placement_planes + self.n_movement_planes + self.n_fight_planes + self.n_extra_planes

        self.action_space_shape = (self.total_action_planes , self.HEIGHT , self.WIDTH)
        self.num_actions     =     self.total_action_planes * self.HEIGHT * self.WIDTH

        # 3D State Representation
        n_terrain_dims = 2
        n_vic_dims = self.N_PLAYERS
        n_reinforcement_dims = self.N_PLAYERS * self.N_UNIT_TYPES
        n_unit_dims = self.N_PLAYERS * self.N_UNIT_TYPES * self.N_UNIT_STATUSES
        n_feature_dims = 3 # turn, phase and player
        total_dims = n_vic_dims + n_reinforcement_dims + n_unit_dims + n_feature_dims # + n_terrain_dims

        self.game_state_shape = (total_dims, self.HEIGHT, self.WIDTH)

        # MCTS support atributes
        self.terminal = False
        self.length = 0
        self.terminal_value = 0
        self.child_policy = []
        self.state_history = []
        self.player_history = []
        self.action_history = []

        self.reset_env()
        self.get_rewards_and_game_env()
        
        return

    def get_board(self):
        return self.board
    
    def getBoardWidth(self):
        return self.WIDTH

    def getBoardHeight(self):
        return self.HEIGHT    

    def getActionRepresentation3D_info(self):
        return self.n_placement_planes, self.n_movement_planes, self.n_fight_planes, self.n_extra_planes, self.total_action_planes, self.action_space_shape

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

    def shallow_clone(self): # TODO improve this function! this is slow and inefficient
        game = SCS_Game(self.reinforcements_by_type[0],self.reinforcements_by_type[1])
        game.reset_env(vp_coordinates=[self.victory_p1, self.victory_p2])
        game.get_rewards_and_game_env()
        for i in range(len(self.action_history)):
            game.step_function(self.action_history[i])

        return game

    def reset_env(self, vp_coordinates=None):
        self.board.clear()
        self.current_player = 1  
        self.current_phase = 0   
        self.current_stage = 0   
        self.current_turn = 1

        if not vp_coordinates:
            x_coords_p1 = np.random.choice(range(self.HEIGHT),size = self.N_VP, replace = False)
            y_coords_p1 = np.random.choice(range(math.floor(self.WIDTH/2)),size = self.N_VP, replace = False)

            for i in range(len(self.victory_p1)):
                self.victory_p1[i][0]=x_coords_p1[i]
                self.victory_p1[i][1]=y_coords_p1[i]
                    
            x_coords_p2 = np.random.choice(range(self.HEIGHT),size = self.N_VP, replace = False)
            y_coords_p2 = np.random.choice(range(math.ceil(self.WIDTH/2), self.WIDTH),size = self.N_VP, replace = False)        

            for i in range(len(self.victory_p2)):
                self.victory_p2[i][0]=x_coords_p2[i]
                self.victory_p2[i][1]=y_coords_p2[i]
        else:
            for i in range(len(vp_coordinates)):
                self.victory_p1 = vp_coordinates[0]
                self.victory_p2 = vp_coordinates[1]


        for p in [0,1]:
            self.available_units[p].clear()
            self.moved_units[p].clear()
            self.atacked_units[p].clear()
            self.reinforcements[p].clear()
            self.reinforcements[p] = copy.copy(self.reinforcements_as_list[p])


        map_choice = 1

        if not self.use_terrain:
            for i in range(self.HEIGHT):
                self.board.append([])
                for j in range(self.WIDTH):
                    self.board[i].append(Tile())

        else:
            if map_choice == 1:
                # Random map
                mountain = Terrain(Atack_modifier=1/2, Defense_modifier=2, Mov_Add_cost=0, Mov_Mult_cost=2, Name="Mountain", image_path="SCS/Images/dirt.jpg")
                plains = Terrain(Atack_modifier=1, Defense_modifier=1, Mov_Add_cost=0, Mov_Mult_cost=1, Name="Plains", image_path="SCS/Images/plains.jpg")
                bush = Terrain(Atack_modifier=2, Defense_modifier=1, Mov_Add_cost=2, Mov_Mult_cost=1, Name="Bush_plains", image_path="SCS/Images/plains_with_bush.jpg")


                self.terrain_types = [mountain, plains, bush]
                probs = [0.05, 0.75, 0.2]
                for i in range(self.HEIGHT):
                    self.board.append([])
                    for j in range(self.WIDTH): 
                        terrain = np.random.choice(self.terrain_types, p=probs)
                        self.board[i].append(Tile(terrain))


        for point in self.victory_p1:
            self.board[point[0]][point[1]].victory = 1

        for point in self.victory_p2:
            self.board[point[0]][point[1]].victory = 2

        # MCTS support atributes
        self.terminal = False
        self.terminal_value = 0
        self.length = 0
        self.child_policy.clear()
        self.state_history.clear()
        self.player_history.clear()
        self.action_history.clear()

        return
    
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

    def possible_actions(self):
        player = self.current_player
        phase = self.current_phase
        size = self.HEIGHT * self.WIDTH
        
        placement_planes = []
        movement_planes = np.zeros((size, self.HEIGHT, self.WIDTH), dtype=np.int32)
        fight_planes = np.zeros((4, self.HEIGHT, self.WIDTH), dtype=np.int32)
        extra_planes = np.zeros((1, self.HEIGHT, self.WIDTH), dtype=np.int32)
        

        if (phase == 0):
            available_types = set(self.reinforcements[player-1])
            for t in range(self.N_UNIT_TYPES):
                if t+1 in available_types:
                    half_collumns = math.floor(self.WIDTH/2)
                    my_half = np.ones((self.HEIGHT, half_collumns), dtype=np.int32)
                    if (half_collumns*2) == self.WIDTH:
                        enemy_half = np.zeros((self.HEIGHT, half_collumns), dtype=np.int32)
                    else:
                        enemy_half = np.zeros((self.HEIGHT, half_collumns+1), dtype=np.int32)
                        
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

            board_shape = (self.HEIGHT, self.WIDTH)
            for unit in self.available_units[player-1]:
                mov = unit.mov_points

                pos_x = unit.tile_x
                pos_y = unit.tile_y

                # Scan possible movements
                while mov > 0:
                    x = mov
                    while x >= 0 :
                        y = mov - x
                        
                        down_x = min(self.HEIGHT-1, pos_x + x)
                        up_x = max(0, pos_x - x )
                        right_y = min(self.WIDTH-1, pos_y + y)
                        left_y = max(0, pos_y - y)
                        
                        if not self.board[down_x][right_y].unit:
                            dest_plane_i = np.ravel_multi_index([down_x, right_y], board_shape) # convert 2D coords into 1D index
                            movement_planes[dest_plane_i][pos_x][pos_y]=1   

                        if not self.board[down_x][left_y].unit:
                            dest_plane_i = np.ravel_multi_index([down_x, left_y], board_shape)
                            movement_planes[dest_plane_i][pos_x][pos_y]=1

                        if not self.board[up_x][right_y].unit:
                            dest_plane_i = np.ravel_multi_index([up_x, right_y], board_shape)
                            movement_planes[dest_plane_i][pos_x][pos_y]=1

                        if not self.board[up_x][left_y].unit:
                            dest_plane_i = np.ravel_multi_index([up_x, left_y], board_shape)
                            movement_planes[dest_plane_i][pos_x][pos_y]=1

                        x-=1
                    mov-=1

                dest_plane_i = np.ravel_multi_index([pos_x, pos_y], board_shape)
                movement_planes[dest_plane_i][pos_x][pos_y] = 1
                    
        if (phase == 2):
            placement_planes = np.zeros((self.N_UNIT_TYPES, self.HEIGHT, self.WIDTH), dtype=np.int32)

            for unit in self.moved_units[player-1]:
                pos_x = unit.tile_x
                pos_y = unit.tile_y
                
                extra_planes[0][pos_x][pos_y] = 1 # no fight action

                enemy_player = (not(player-1)) + 1 
                _ , enemy_dir = self.check_adjacent_units(unit, enemy_player)
                for direction in enemy_dir:
                    fight_planes[direction][pos_x][pos_y] = 1


        planes_list = [placement_planes, movement_planes, fight_planes, extra_planes]
        valid_actions_mask = np.concatenate(planes_list)
        return valid_actions_mask

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

    def check_adjacent_units(self, unit, enemy):
        x = unit.tile_x
        y = unit.tile_y
        
        if(x+1) == self.HEIGHT:
            down = None
        else:
            down = self.board[x+1][y].unit

        if(x-1) == -1:
            up = None
        else:
            up = self.board[x-1][y].unit

        if (y+1) == self.WIDTH:
            right = None
        else:
            right = self.board[x][y+1].unit
        
        if (y-1) == -1:
            left = None
        else:
            left = self.board[x][y-1].unit
            
        adjacent_units = []
        enemy_directions = []
        if(up):
            if (up.player==enemy):
                adjacent_units.append(up)
                enemy_directions.append(0)
        if(down):
            if (down.player==enemy):
                adjacent_units.append(down)
                enemy_directions.append(1)
        if(right):
            if (right.player==enemy):
                adjacent_units.append(right)
                enemy_directions.append(2)
        if(left):
            if (left.player==enemy):
                adjacent_units.append(left)
                enemy_directions.append(3)

        return adjacent_units , enemy_directions

    def resolve_combat(self, unit, enemy):
        unit_x = unit.tile_x
        unit_y = unit.tile_y
        enemy_x = enemy.tile_x
        enemy_y = enemy.tile_y
        player = unit.player
        e = (not(player-1)) + 1 #enemy player
        
        remaining_enemy_defense = enemy.defense - unit.atack
        remaining_ally_defense = unit.defense - enemy.atack

        if remaining_enemy_defense <= 0:
            if remaining_ally_defense<=0:
                self.board[enemy_x][enemy_y].unit = None
                self.moved_units[player-1].remove(unit)
            else:
                unit.edit_defense(remaining_ally_defense)
                self.board[enemy_x][enemy_y].unit = unit
                

                # Take the enemy position
                self.moved_units[player-1].remove(unit)
                unit.move_to(enemy_x, enemy_y)
                self.atacked_units[player-1].append(unit)

            if e==1:
                self.atacked_units[e-1].remove(enemy)
            elif e==2:
                self.available_units[e-1].remove(enemy)
            else:
                print("Problem with enemy player.Exiting.")
                exit()

            self.board[unit_x][unit_y].unit = None

        else:
            if remaining_ally_defense<=0:
                enemy.edit_defense(remaining_enemy_defense)
                self.moved_units[player-1].remove(unit)
                self.board[unit_x][unit_y].unit = None
                del unit 
            else:
                unit.edit_defense(remaining_ally_defense)
                enemy.edit_defense(remaining_enemy_defense)

        
        return

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

        elif (act == 3): # no fighting
            string = "Fighting phase: Unit at (" + str(start[0]+1) + "," + str(start[1]+1) + ") " + "chose not to fight"

        else:
            string = "Unknown action..."

        #print(string)
        return string

    def parse_action(self, action_coords):
        act = None
        unit_type = None
        start = (None, None)
        dest = (None, None)

        size = self.HEIGHT*self.WIDTH
        if action_coords[0] < self.N_UNIT_TYPES:
            act = 0
            unit_type = action_coords[0] + 1  # 1 -> soldier | 2 -> tank
            start = (action_coords[1], action_coords[2])
        elif action_coords[0] < self.N_UNIT_TYPES + size:
            act = 1
            start = (action_coords[1], action_coords[2])
            dest = np.unravel_index(action_coords[0]-self.N_UNIT_TYPES, (self.HEIGHT, self.WIDTH))
        elif action_coords[0] < self.N_UNIT_TYPES + size + 4:
            act = 2
            x = action_coords[1]
            y = action_coords[2]
            start = (x, y)
            dir_i = action_coords[0] - (self.N_UNIT_TYPES + size)

            # UP(0) DOWN(1) RIGHT(2) LEFT(3)
            if dir_i == 0:
                dest= (x-1,y)
            elif dir_i == 1:
                dest= (x+1,y)
            elif dir_i == 2:
                dest= (x,y+1)
            elif dir_i == 3:
                dest= (x,y-1)
            else:
                print("Problem parsing action...Exiting")
                exit()

        elif action_coords[0] < self.N_UNIT_TYPES + size + 5:
            act = 3
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
            self.available_units[self.current_player-1].remove(unit)
            if start != dest:
                unit.move_to(dest[0],dest[1])
                self.board[start[0]][start[1]].unit = None
                self.board[dest[0]][dest[1]].place_unit(unit)

            self.moved_units[self.current_player-1].append(unit)

        elif (act == 2): # fighting
            my_unit = self.board[start[0]][start[1]].unit
            enemy_unit = self.board[dest[0]][dest[1]].unit
            self.resolve_combat(my_unit, enemy_unit)

        elif (act == 3): # no fighting
            my_unit = self.board[start[0]][start[1]].unit
            self.atacked_units[self.current_player-1].append(my_unit)
            self.moved_units[self.current_player-1].remove(my_unit)

        else:
            print("Unknown action. Exiting...")
            exit()

        return

    def check_final_rewards(self):
        rewards = [0,0]
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
        p1_reward = point_diff/self.N_VP
        p2_reward = -p1_reward

        final_value = p1_reward
        rewards = [p1_reward, p2_reward]
        return rewards , final_value

    def get_rewards_and_game_env(self):

        done = False
        previous_player = self.current_player
        previous_stage = self.current_stage
        stage = previous_stage
        
        while True:
            if stage == 0 and self.reinforcements[0] == []:
                stage+=1
                continue
            if stage == 1 and self.reinforcements[1] == []:
                stage+=1
                continue
            if stage == 2 and self.available_units[0] == []:
                stage+=1
                continue
            if stage == 3 and self.moved_units[0] == []:
                stage+=1
                continue
            if stage == 4 and self.available_units[1] == []:
                stage+=1
                continue
            if stage == 5 and self.moved_units[1] == []:
                if self.current_turn+1 > self.TURNS:
                    done = True
                    self.terminal = True
                    break
                self.current_turn+=1
                stage=0
                self.new_turn()
                continue
            break

        rewards=[0,0]
        reward = 0

        if(done):
            rewards, final_value = self.check_final_rewards()
            self.terminal_value = final_value
        #else:
        #    rewards = self.intermidiate_rewards()
        '''
        if previous_player == 1:
            reward = rewards[0]

        elif previous_player == 2:
            reward = rewards[1]

        '''
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

        return rewards, done

    def new_turn(self):
        self.available_units = self.atacked_units.copy()
        self.atacked_units = [[], []]

        return  

    def intermidiate_rewards(self):
        rewards = [0,0]

        #get rewarded for holding an enemy point
        for point in self.victory_p2:
            vic_p2 = self.board[point[0]][point[1]]
            if vic_p2.unit:
                if(vic_p2.unit.player==1):
                    rewards[0]+=0.5
        
        
        for point in self.victory_p1:
            vic_p1 = self.board[point[0]][point[1]]
            if vic_p1.unit:
                if(vic_p1.unit.player==2):
                    rewards[1]+=0.5

        return rewards.copy()

    def generate_state_image(self):

        p1_victory = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)
        p2_victory = np.zeros((self.HEIGHT, self.WIDTH), dtype=np.int32)

        p_units = [[],[]]

        p1_reinforcements = [[] for u in range(self.N_UNIT_TYPES)]
        p2_reinforcements = [[] for u in range(self.N_UNIT_TYPES)]

        p1_reinforcement_counts = list(Counter(self.reinforcements[0]).values())
        p2_reinforcement_counts = list(Counter(self.reinforcements[1]).values())


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


        for v in self.victory_p1:
            x = v[0]
            y = v[1]
            p1_victory[x][y] = 1

        for v in self.victory_p2:
            x = v[0]
            y = v[1]
            p2_victory[x][y] = 1

        for p in [0,1]: # for each player check each unit status
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

        player_plane = np.ones((self.HEIGHT,self.WIDTH), dtype=np.int32)
        if self.current_player == 2:
            player_plane.fill(-1)

        phase = self.current_phase
        turn = self.current_turn

        player_plane = torch.unsqueeze(torch.as_tensor(player_plane,dtype=torch.float32), 0)

        p1_victory = torch.unsqueeze(torch.as_tensor(p1_victory, dtype=torch.float32), 0)
        p2_victory = torch.unsqueeze(torch.as_tensor(p2_victory, dtype=torch.float32), 0)

        p1_reinforcements = torch.stack(p1_reinforcements, dim=0)
        p2_reinforcements = torch.stack(p2_reinforcements, dim=0)

        state_phase = torch.full((self.HEIGHT, self.WIDTH), phase, dtype=torch.float32)
        state_phase = torch.unsqueeze(state_phase, 0)

        state_turn = torch.full((self.HEIGHT, self.WIDTH), turn, dtype=torch.float32)
        state_turn = torch.unsqueeze(state_turn, 0)

        p1_units = torch.as_tensor(np.array(p_units[0]), dtype=torch.float32)
        p2_units = torch.as_tensor(np.array(p_units[1]), dtype=torch.float32)

        
        stack_list = (p1_victory, p2_victory, p1_units, p2_units, p1_reinforcements, p2_reinforcements, state_phase, state_turn, player_plane) # add terrain
        new_state = torch.concat(stack_list, dim=0)

        state_image = torch.unsqueeze(new_state, 0)# add batch size to the dimensions
        return state_image

    def step_function(self, action_coords):
        reward = None
        done = False
        self.play_action(action_coords)
        self.action_history.append(action_coords)
        self.length += 1
        reward, done = self.get_rewards_and_game_env()
        return reward, done
    
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

    def store_state(self, state):
        self.state_history.append(state)
        return

    def store_player(self, player):
        self.player_history.append(player)
        return
    
    def get_name(self):
        return "SCS"
# ------------------------------------------------------------ #
# ---------------------- USER FUNCTIONS ---------------------- #
# ------------------------------------------------------------ #

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
                    (((pos_x == (x-1)) and (pos_y == (y-1))) or self.board[x-1][y-1].unit == None)):
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
        # WARNING: User games do not maintain an updated game enviorment ( they are just for fun ^^ )

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
    


