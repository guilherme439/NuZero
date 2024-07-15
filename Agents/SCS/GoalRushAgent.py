from Games.SCS.SCS_Game import SCS_Game

import heapq
import numpy as np

from Agents.Agent import Agent

class GoalRushAgent(Agent):
    ''' Tries to move each unit towards the closest victory point. 
        Atacks enemies if they are in the path to the goal. '''

    def __init__(self):
        self.graph = None
        return

    def choose_action(self, game):
        if not isinstance(game, SCS_Game):
            print("GoalRushAgent can only be used for SCS Games")
            exit()
        if self.graph is None:
            print("GoalRushAgent requires new_game() to be called before each game.")
            exit()

        player = game.current_player
        if game.current_stage in game.reinforcement_stages():
            # This agent places reinforcements at random
            possible_actions = game.possible_actions().flatten()
            valids = sum(possible_actions)
            probs = possible_actions/sum(possible_actions)
            action_i = np.random.choice(game.num_actions, p=probs)
            return game.get_action_coords(action_i)


        if game.current_stage in game.movement_stages():
            # If it is the movement phase then, there is a unit to move
            unit = game.available_units[player-1][0]
            unit_position = unit.position
            unit_tile = game.get_tile(unit_position)
            unit_stacking = unit_tile.get_stacking_level(unit)
            opponent = game.opponent(player)
            vic_points = game.victory_points[opponent-1]

            distances, previous_nodes = self.dijkstra(unit_position)

            # Check which victory point is closest and move towards it
            closest_vp = None
            min_distance = np.inf
            for node, distance in distances.items():
                if node in vic_points:
                    if distance < min_distance:
                        min_distance = distance
                        closest_vp = node
                        

            path = self.path_to(closest_vp, previous_nodes)
            if len(path) > 0:
                position_to_move = path[0]
                action_i, action_coords = game.get_movement_action(unit_position, unit_stacking, position_to_move)

                possible_actions = game.possible_actions().flatten()
                if possible_actions[action_i]:
                    return action_coords
            
            # if it is not possible to move towards the goal or we are already there, don't move 
            _, action_coords = game.get_movement_action(unit_position, unit_stacking, unit_position)
            return action_coords
        
        elif game.current_stage in game.choosing_target_stages():
            player_units = game.moved_units[player-1]
            opponent = game.opponent(player)
            vic_points = game.victory_points[opponent-1]

            for unit in player_units:
                unit_position = unit.position
                # Check all the paths to closest victory points,
                # if there is an enemy in any of the paths, target that enemy
                distances, previous_nodes = self.dijkstra(unit_position)
                
                closest_vp = None
                min_distance = np.inf
                for node, distance in distances.items():
                    if node in vic_points:
                        if distance < min_distance:
                            min_distance = distance
                            closest_vp = node

                path = self.path_to(closest_vp, previous_nodes)
                if len(path) > 0:
                    position_to_move = path[0]
                    destination_tile = game.get_tile(position_to_move)
                    tile_owner = destination_tile.player
                    if tile_owner == opponent:
                        action_i, action_coords = game.get_target_action(position_to_move)
                        return action_coords
            
            # If no enemy is found,
            # or we are already at the victory point,
            # tell the first unit to skip combat
            first_unit = player_units[0]
            position = first_unit.position
            stacking = game.get_tile(first_unit.position).get_stacking_level(first_unit)
            action_i, action_coords = game.get_skip_combat_action(position, stacking)
            return action_coords
        
        elif game.current_stage in game.choosing_attackers_stages():
            possible_actions = game.possible_actions().flatten()
            action_i, action_coords = game.get_confirm_attack_action()
            valids = sum(possible_actions)

            # Choose attackers at random until the only possible action is to confirm the attack
            if (valids == 1) and (possible_actions[action_i] == 1):                 
                return action_coords
            else:
                possible_actions[action_i] = 0
                probs = possible_actions/sum(possible_actions)
                action_i = np.random.choice(game.num_actions, p=probs)
                return game.get_action_coords(action_i)
   
    def create_graph(self, game):
        '''Creates the graph for Dijkstra's Algorithm as a dict of dicts'''
        board = game.get_board()
        graph = {}
        for row in board:
            for tile in row:
                node = {}
                tile_position = tile.position
                adjacent_tiles = game.check_tiles(tile_position)
                for neighbour in adjacent_tiles:
                    if neighbour is not None:
                        neighbour_position = neighbour.position
                        cost = neighbour.get_terrain().cost
                        node[neighbour_position] = cost

                graph[tile_position] = node
        
        return graph
    
    def dijkstra(self, start):
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        previous_nodes = {node: None for node in self.graph}
        queue = [(0, start)]

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in self.graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        return distances, previous_nodes
    
    def path_to(self, destination, previous_nodes):
        path = []
        while destination:
            path.append(destination)
            destination = previous_nodes[destination]
        
        path.pop(-1)
        path.reverse()
        return path
    
    def name(self):
        return "GoalRush Agent"
    
    def new_game(self, game, *args, **kwargs):
        if game is None:
            print("WARNING: GoalRushAgent.new_game() without a \"game\" argument.")
            print("When called this way the internal graph won't be updated,\n \
                   and the Agent will likely not trace the fastest paths to the VPs.")
        
        else:
            self.graph = self.create_graph(game)