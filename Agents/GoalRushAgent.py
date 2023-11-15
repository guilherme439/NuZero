from SCS.SCS_Game import SCS_Game
import heapq

    

class GoalRushAgent():
    ''' Tries to move each unit towards the closest victory point. 
        Atacks enemies if they are in the path to the goal. '''

    def __init__(self, game):
        self.game = game
        self.graph = self.create_graph()
        return

    def create_graph(self):
        '''Creates the graph for Dijkstra's Algorithm as a dict of dicts'''
        board = self.game.get_board()
        graph = {}
        for row in board:
            for tile in row:
                node = {}
                tile_position = tile.position
                adjacent_tiles = self.game.check_tiles(tile_position)
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
    
    def shortest_path_to(self, destination, previous_nodes):
        path = []
        while destination:
            path.append(destination)
            destination = previous_nodes[destination]
        return path[::-1]
    
    def take_action(self, game):
        if not isinstance(game, SCS_Game):
            print("GoalRushAgent can only be used for SCS Games")
            exit()

        player = game.current_player
        if game.current_stage in (1,5):
            unit = game.available_units[player-1][0]
            unit_position = unit.position
    
    

    def closest_victory_point(self, unit):
        player = unit.player
        opponent = self.game.opponent(player)
        unit_position = unit.position

        vps = self.game.victory_points[opponent]

        for point in vps:
            distance = 0

    def distance(self, point1, point2):
        return