import pygame
import time
import ray
import math

from enum import Enum
from .Soldier import Soldier

import sys
sys.path.append("..")
from RemoteStorage import RemoteStorage

class Color(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    ORANGE = (200, 100, 0)
    RED = (200, 0, 0)
    BROWN = (90, 50, 0)
    GREEN = (45, 110, 10)
    BAD_PINK = (255, 0, 255)

    def rgb(self):
        return self.value


@ray.remote
class SCS_Renderer():

    def __init__(self, remote_storage=None):
        self.game_storage = remote_storage

        # Set the width and height of the output window, in pixels
        self.WINDOW_WIDTH = 1200
        self.WINDOW_HEIGHT = 1000

    # Passively render a game while it is being played, using a remote storage for communication
    def render(self):

        pygame.init()

        
        # A remote game storage is used to update the game being displayed
        game = ray.get(self.game_storage.get_item.remote())

        
        # Set up the drawing window
        screen = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])

        time.sleep(0.2)
        # Run until user closes window
        running=True
        while running:

            game = ray.get(self.game_storage.get_item.remote())
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False
            if game.is_terminal():
                running=False

            # Fill the background with white
            screen.fill(Color.WHITE.rgb())

            self.render_board_hexagons(screen, game)
            #self.render_board_squares(screen, game)

            text = "SCS Board live rendering!"
            if len(game.action_history) > 0:
                last_action = game.action_history[-1]
                text = game.string_action(last_action)


            text_font = pygame.font.SysFont("meera", 50)
            text_block = text_font.render(text, True, Color.RED.rgb())
            text_rect = text_block.get_rect(center=(self.WINDOW_WIDTH/2, 50))
            screen.blit(text_block, text_rect)

            turn_text = "Turn: " + str(game.current_turn)
            turn_font = pygame.font.SysFont("meera", 30)
            turn_block = turn_font.render(turn_text, True, Color.BLACK.rgb())
            screen.blit(turn_block, (30, 30))

            # Update de full display
            pygame.display.flip()

            # Limit fps
            time.sleep(0.6)
        
        # Done! Time to quit.
        pygame.quit()
        return
    
    # Interactively render an already played game using arrow keys
    def analyse(self, game):
        
        pygame.init()

        render_game = game.clone() # scratch game for rendering
        
        # Set up the drawing window
        screen = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])

        action_index = 0
        time.sleep(0.1)
        # Run until user closes window
        running=True
        while running:
            
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        running=False
                    
                    case pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT:
                            if action_index < game.get_length():
                                action_index +=1

                        elif event.key == pygame.K_LEFT:
                            if action_index > 0:
                                action_index -=1

            render_game.reset_env()
            for i in range(action_index):
                action = game.action_history[i]
                render_game.step_function(action)

            # Fill the background with white

            screen.fill(Color.WHITE.rgb())
    
            self.render_board(screen, render_game)


            action_text = "SCS Analisis board!"
            if len(render_game.action_history) > 0:
                last_action = render_game.action_history[-1]
                action_text = render_game.string_action(last_action)

            winner_text = ":-)"
            if len(render_game.action_history) == len(game.action_history):
                winner = game.check_winner()
                if winner == 0:
                    winner_text = "Draw!"
                else:
                    winner_text = "Player " + str(winner) + " won!"

            action_number_text = "Actions played: " + str(action_index)

            action_font = pygame.font.SysFont("meera", 40)
            action_block = action_font.render(action_text, True, Color.RED.rgb())
            action_rect = action_block.get_rect(center=(self.WINDOW_WIDTH/2, 50))
            screen.blit(action_block, action_rect)


            turn_text = "Turn: " + str(render_game.current_turn)
            turn_font = pygame.font.SysFont("rachana", 25)
            turn_block = turn_font.render(turn_text, True, Color.BLACK.rgb())
            turn_rect = turn_block.get_rect(topleft=(5, 5))
            screen.blit(turn_block, turn_rect)

            action_number_font = pygame.font.SysFont('notosansmono', 20)
            action_number_block = action_number_font.render(action_number_text, True, Color.GREEN.rgb())
            action_number_rect = action_number_block.get_rect(bottomleft=(5, self.WINDOW_HEIGHT-5))
            screen.blit(action_number_block, action_number_rect)

            winner_font = pygame.font.SysFont('notosansmonocjkkr', 20)
            winner_font.set_bold(True)
            winner_block = winner_font.render(winner_text, True, Color.ORANGE.rgb())
            winner_rect = winner_block.get_rect(bottomright=(self.WINDOW_WIDTH-5, self.WINDOW_HEIGHT-5))
            screen.blit(winner_block, winner_rect)

            # Update de full display
            pygame.display.flip()

            # Limit fps
            time.sleep(0.2)
        
        # Done! Time to quit.
        pygame.quit()
        return
    
    def render_board_squares(self, screen, game, debug=[]):

        if len(debug) > 0:
            values, positions = list(zip(*debug))

        GAME_HEIGHT = game.getBoardHeight()
        GAME_WIDTH = game.getBoardWidth()

        # Draw the board
        board_top_offset = math.floor(0.15*self.WINDOW_HEIGHT)
        board_bottom_offset = math.floor(0.05*self.WINDOW_HEIGHT)

        board_height = (self.WINDOW_HEIGHT - board_top_offset - board_bottom_offset)
        board_height = board_height - (board_height%GAME_HEIGHT) # make sure the board height is divisible by the number of tiles

        board_width = board_height

        tile_height = board_height//GAME_HEIGHT
        tile_width = tile_height
        
        # values in pixels
        tile_border_width = 2
        board_border_width = 8
        
        numbers_gap = 25

        board_center = (self.WINDOW_WIDTH//2, board_top_offset + board_height/2)
        
        x_offset = board_center[0] - board_width//2
        y_offset = board_center[1] - board_height//2
        

        board_position = (x_offset-board_border_width, y_offset-board_border_width)
        board_dimensions = (board_width+(2*board_border_width), board_height+(2*board_border_width))
        board_border = pygame.Rect(board_position, board_dimensions)
        pygame.draw.rect(screen, Color.BROWN.rgb(), board_border, board_border_width)


        board = game.get_board()
        for i in range(GAME_HEIGHT):
            
            # BOARD NUMBERS
            number_font = pygame.font.SysFont("uroob", 30)
            number_block = number_font.render(str(i+1), True, Color.BLACK.rgb())
            number_rect = number_block.get_rect(center=(board_position[0] - numbers_gap, board_position[1] + tile_height/2 + (tile_height)*i))
            screen.blit(number_block, number_rect)

            for j in range(GAME_WIDTH):

                # x goes left and right
                # j goes left and right
                # y goes up and down
                # i goes up and down

                # BOARD NUMBERS
                if i==0:
                    number_font = pygame.font.SysFont("uroob", 30)
                    number_block = number_font.render(str(j+1), True, Color.BLACK.rgb())
                    number_rect = number_block.get_rect(center=(board_position[0] + tile_width/2 + (tile_width)*j, board_position[1] - numbers_gap))
                    screen.blit(number_block, number_rect)


                # TILES
                x_position = ((tile_width)*j)+x_offset
                y_position = ((tile_height)*i)+y_offset
                tile_position = (x_position, y_position)
                tile_dimensions = (tile_height, tile_width)
                tile_rect = pygame.Rect(tile_position, tile_dimensions)
                pygame.draw.rect(screen, Color.BLACK.rgb(), tile_rect, tile_border_width)

                tile = board[i][j]


                # TERRAIN
                terrain = tile.get_terrain()                
                if terrain:
                    terrain_image = pygame.image.load(terrain.get_image_path())

                    terrain_dimensions = (tile_width-(2*tile_border_width), tile_height-(2*tile_border_width))
                    terrain_position = (tile_position[0]+tile_border_width, tile_position[1]+tile_border_width)
                    terrain_surface = pygame.transform.scale(terrain_image, terrain_dimensions)
        
                    screen.blit(terrain_surface, terrain_position)

                # DEBUG INFO
                if len(debug) > 0:
                    if (i,j) in positions:
                        idx = positions.index((i,j))
                        value = values[idx]
                        value_text = format(value, '.3')
                        value_font = pygame.font.SysFont('notosansmonocjkkr', 25)
                        value_font.set_bold(True)
                        value_block = value_font.render(value_text, True, Color.BLACK.rgb())
                        value_text_position = (tile_position[0] + tile_height/2, tile_position[1] + tile_width/2)
                        value_rect = value_block.get_rect(center=value_text_position)
                        screen.blit(value_block, value_rect)

                # VICTORY POINTS
                vp = tile.victory
                p1_path = "SCS/Images/blue_star.png"
                p2_path = "SCS/Images/red_star.png"
                if vp != 0:
                    if vp == 1:
                        star_image = pygame.image.load(p1_path)
                    elif vp == 2:
                        star_image = pygame.image.load(p2_path)

                    # As percentage of tile size
                    star_scale = 0.2
                    star_margin = 0.1

                    star_dimensions = (star_scale*tile_dimensions[0], star_scale*tile_dimensions[1])
                    star_x_offset = (1-(star_scale+star_margin))*tile_dimensions[0]
                    star_y_offset = star_margin*tile_dimensions[1]
                    star_position = (tile_position[0] + star_x_offset, tile_position[1] + star_y_offset)
                    star_surface = pygame.transform.scale(star_image, star_dimensions)
        
                    screen.blit(star_surface, star_position)

                # UNITS
                unit = tile.unit
                if unit:
                    unit_scale = 0.75
                    unit_image = pygame.image.load(unit.get_image_path())

                    unit_dimensions = (unit_scale*tile_dimensions[0], unit_scale*tile_dimensions[1])

                    unit_x_offset = (tile_dimensions[0]-unit_dimensions[0])//2
                    unit_y_offset = (tile_dimensions[1]-unit_dimensions[1])//2
                    unit_position = (tile_position[0] + unit_x_offset, tile_position[1] + unit_y_offset)
                    unit_surface = pygame.transform.scale(unit_image, unit_dimensions)

                    screen.blit(unit_surface, unit_position)

    def draw_hexagon(self, surface, color, radius, position, width=0):
        n = 6
        r = radius
        x, y = position
        rectangle = pygame.draw.polygon(surface, color, [
            (x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n))
            for i in range(n)
        ], width)

        return rectangle

    def render_board_hexagons(self, screen, game, debug=[]):

        if len(debug) > 0:
            values, positions = list(zip(*debug))

        height_to_width_ratio = 1.1547005
        width_to_height_ratio = 0.8660254

        GAME_ROWS = game.getBoardRows()
        GAME_COLS = game.getBoardColumns()

        # values in pixels
        tile_border_thickness = 3
        board_border_thickness = 8
        if tile_border_thickness % 2 == 0:
            outter_tile_border_thickness = (tile_border_thickness / 2) - 1
        else:
            outter_tile_border_thickness = (tile_border_thickness // 2)

        numbers_gap = 25

        # Dimensions
        star_scale = 0.3 # as a percentage of hexagon size

        unit_scale = 0.6 # as a percentage of tile size

        # Board sizes
        board_top_gap = math.floor(0.15*self.WINDOW_HEIGHT)
        board_bottom_gap = math.floor(0.05*self.WINDOW_HEIGHT)

        board_right_gap = math.floor(0.05*self.WINDOW_HEIGHT)
        board_left_gap = board_right_gap

        board_height = (self.WINDOW_HEIGHT - board_top_gap - board_bottom_gap)
        board_height = board_height - (board_height%GAME_ROWS) # make sure the board height is divisible by the number of rows

        board_width = self.WINDOW_WIDTH - board_left_gap - board_right_gap
        board_width = board_width - (board_width%GAME_COLS)

        # Find the max size of each hexagon based on the available space and the number of rows and cols
        horizontal_number_of_sides = 1.5 * GAME_COLS + 0.5
        
        vertical_number_of_short_sides =  (GAME_ROWS*2) + 1
        vertical_number_of_sides = vertical_number_of_short_sides * width_to_height_ratio

        width_based_side = board_width // horizontal_number_of_sides
        height_based_side = board_height // vertical_number_of_sides

        hexagon_side = min(width_based_side, height_based_side)
        hexagon_short_side = hexagon_side * width_to_height_ratio
        radius = hexagon_side
        
        border_rectangle_width = hexagon_side * horizontal_number_of_sides + (board_border_thickness * 2)
        border_rectangle_height = hexagon_side * vertical_number_of_sides + (board_border_thickness * 2)
        border_dimensions = (border_rectangle_width, border_rectangle_height)
        
        board_center = (board_left_gap + board_width//2, board_top_gap + board_height//2)
        border_rectangle = pygame.Rect((0,0), border_dimensions)
        border_rectangle.center = board_center

        board_x = border_rectangle.x
        board_y = border_rectangle.y

        x_offset = board_x + hexagon_side + board_border_thickness
        y_offset = board_y + hexagon_short_side + board_border_thickness
        

        pygame.draw.rect(screen, Color.BROWN.rgb(), border_rectangle, board_border_thickness)
        
        board = game.get_board()
        for i in range(GAME_ROWS):
            for j in range(GAME_COLS):

                # x goes left and right
                # j goes left and right
                # y goes up and down
                # i goes up and down

                odd_col = j % 2

                center_x = x_offset + j*((3/2*hexagon_side))
                center_y = y_offset + i*((hexagon_short_side*2))
                if odd_col:
                    center_y += hexagon_short_side


                # BOARD NUMBERS
                if j==0:
                    number_font = pygame.font.SysFont("uroob", 30)
                    number_block = number_font.render(str(i+1), True, Color.BLACK.rgb())
                    number_rect = number_block.get_rect(center=(board_x - numbers_gap, center_y))
                    screen.blit(number_block, number_rect)
                if i==0:
                    number_font = pygame.font.SysFont("uroob", 30)
                    number_block = number_font.render(str(j+1), True, Color.BLACK.rgb())
                    number_rect = number_block.get_rect(center=(center_x, board_y - numbers_gap))
                    screen.blit(number_block, number_rect)

                
                # TILES
                tile = board[i][j]
                tile_radius = radius-(outter_tile_border_thickness*height_to_width_ratio)
                tile_center = (center_x, center_y)
                tile_rect_dims = ((2*hexagon_side, hexagon_short_side*2)) 

                # TERRAIN
                terrain = tile.get_terrain()   
                if terrain:   
                    hexagon_surface = pygame.Surface(tile_rect_dims)
                    hexagon_surface.fill(Color.BAD_PINK.rgb())
                    terrain_radius = tile_radius - (outter_tile_border_thickness*height_to_width_ratio) + 1 # We add 1 to slighylt overlap the image behind the border
                    self.draw_hexagon(hexagon_surface, Color.WHITE.rgb(), terrain_radius, hexagon_surface.get_rect().center, width=0)


                    terrain_image = pygame.image.load(terrain.get_image_path())
                    terrain_surface = pygame.transform.scale(terrain_image, tile_rect_dims)
                    pygame.transform.threshold(terrain_surface, hexagon_surface, Color.BAD_PINK.rgb(), inverse_set=True, set_color=Color.BAD_PINK.rgb())
                    terrain_surface.set_colorkey(Color.BAD_PINK.rgb())
                    terrain_rect = terrain_surface.get_rect(center=(center_x, center_y))

                    screen.blit(terrain_surface, terrain_rect)
                    self.draw_hexagon(screen, Color.BLACK.rgb(), tile_radius, (center_x, center_y), width=tile_border_thickness)

                tile_rect = self.draw_hexagon(screen, Color.BLACK.rgb(), tile_radius, (center_x, center_y), width=tile_border_thickness)
                # Delay tile rendering util after  terrain rendering

                # DEBUG INFO
                if len(debug) > 0:
                    if (i,j) in positions:
                        idx = positions.index((i,j))
                        value = values[idx]
                        value_text = format(value, '.3')
                        value_font = pygame.font.SysFont('notosansmonocjkkr', 25)
                        value_font.set_bold(True)
                        value_block = value_font.render(value_text, True, Color.BLACK.rgb())
                        value_text_position = (tile_rect.center)
                        value_rect = value_block.get_rect(center=value_text_position)
                        screen.blit(value_block, value_rect)

                # VICTORY POINTS
                vp = tile.victory
                p1_path = "SCS/Images/blue_star.png"
                p2_path = "SCS/Images/red_star.png"
                if vp != 0:
                    if vp == 1:
                        star_image = pygame.image.load(p1_path)
                    elif vp == 2:
                        star_image = pygame.image.load(p2_path)

                    star_dimensions = (star_scale*hexagon_side, star_scale*hexagon_side)
                    star_surface = pygame.transform.scale(star_image, star_dimensions)
                    (midtop_x, midtop_y) = tile_rect.midtop
                    star_x_offset = hexagon_side * 0.4
                    star_y_offset = hexagon_side * 0.25
                    star_position = (midtop_x + star_x_offset, midtop_y + star_y_offset)
                    star_rect = star_surface.get_rect(center=star_position)
                    screen.blit(star_surface, star_rect)

                # UNITS
                unit = tile.unit
                if unit:
                    unit_image = pygame.image.load(unit.get_image_path())
                    image_height = unit_image.get_rect().h
                    tile_size_ratio = tile_rect_dims[0]/image_height

                    unit_surface = pygame.transform.scale_by(unit_image, tile_size_ratio)
                    unit_surface = pygame.transform.scale_by(unit_surface, unit_scale)
                    unit_rect = unit_surface.get_rect(center=tile_center)                   
                    screen.blit(unit_surface, unit_rect)

    def debug_value(self, move_num, base_game, nn, recurrent_iterations=2):
        games_list = []
        
        base_game.reinforcements = [[],[]]
        for i in range(base_game.HEIGHT):
            for j in range(base_game.WIDTH):
                tile = base_game.board[i][j]
                scratch_game = base_game.shallow_clone()
                scratch_tile = scratch_game.board[tile.x][tile.y]
                position = (tile.x, tile.y)
                new_unit = Soldier(1, tile.x, tile.y)
                scratch_game.available_units[0].append(new_unit)
                scratch_tile.place_unit(new_unit)
                scratch_game.update_game_env()
                games_list.append((scratch_game, position))


        no_move_index = (base_game.placement_planes + base_game.movement_planes + base_game.fight_planes + base_game.no_move_planes) - 1

        values_list = []
        for i in range(len(games_list)):
            pair = games_list[i]
            game = pair[0]
            position = pair[1]

            for j in range(move_num):
                action_coords = (no_move_index, position[0], position[1])
                game.step_function(action_coords)

            state_image = game.generate_state_image()
            _, value_pred = nn.inference(state_image, False, recurrent_iterations)

            values_list.append((value_pred.item(), position))

        
    
        pygame.init()

        render_game = base_game.clone() # scratch game for rendering
        
        # Set up the drawing window
        screen = pygame.display.set_mode([self.WINDOW_WIDTH, self.WINDOW_HEIGHT])

        time.sleep(0.1)
        # Run until user closes window
        running=True
        while running:
            
            for event in pygame.event.get():
                match event.type:
                    case pygame.QUIT:
                        running=False
                    

            # Fill the background with white

            screen.fill(Color.WHITE.rgb())
    
            self.render_board(screen, render_game, debug=values_list)


            title_text = "SCS Value Debug"
            title_font = pygame.font.SysFont("meera", 40)
            title_block = title_font.render(title_text, True, Color.RED.rgb())
            title_rect = title_block.get_rect(center=(self.WINDOW_WIDTH/2, 50))
            screen.blit(title_block, title_rect)

            # Update de full display
            pygame.display.flip()

            # Limit fps
            time.sleep(0.4)
        
        # Done! Time to quit.
        pygame.quit()



        return 
# ------------------------------------------------------ #
# ----------------------- FONTS ------------------------ #
# ------------------------------------------------------ #

    def all_fonts(self):
        fonts = [
        'tlwgtypo', 'dejavuserif', 'urwbookman', 'kalapi', 'rekha', 'tlwgtypewriter', 'dejavusansmono', 'ubuntumono',
        'rachana', 'liberationmono', 'pottisreeramulu', 'anjalioldlipi', 'suravaram', 'notoserifcjksc', 'keraleeyam', 'c059',
        'garuda', 'nimbusmonops', 'notosansmono', 'notoserifcjktc', 'freesans', 'p052', 'liberationsansnarrow', 'kacstfarsi',
        'padaukbook', 'dejavusans','nimbussans', 'rasa', 'liberationsans', 'nimbussansnarrow', 'padmaa', 'notoserifcjkjp',
        'notoserifcjkhk', 'notoserifcjkkr', 'freeserif', 'abyssinicasil', 'uroob', 'yrsa', 'mrykacstqurn', 'tlwgtypist', 'peddana',
        'kacstone', 'freemono', 'gayathri', 'notosanscjkjp', 'notosanscjkhk', 'notosanscjkkr', 'loma', 'liberationserif',
        'padauk', 'kacstdigital', 'ubuntu', 'kacstpen', 'ponnala', 'notosanscjksc', 'laksaman', 'chilanka', 'notosanscjktc',
        'kinnari', 'lohitgurmukhi', 'tlwgmono', 'ramaraja', 'mitra', 'waree', 'sarai', 'manjari', 'umpush', 'z003', 'urwgothic',
        'sawasdee', 'lohitbengali', 'kacstscreen', 'kacstart', 'saab', 'samyaktamil', 'lohitgujarati', 'd050000l', 'lohitassamese',
        'timmana', 'raviprakash', 'norasi', 'purisa', 'nimbusroman', 'khmeros', 'opensymbol', 'gidugu', 'lohitdevanagari',
        'kalimati', 'droidsansfallback', 'khmerossystem', 'lohittelugu', 'ramabhadra', 'nats', 'lohitodia', 'karumbi', 'phetsarathot',
        'kacstdecorative', 'lklug', 'ani', 'lakkireddy', 'lohittamilclassical', 'tenaliramakrishna', 'jamrul','pagul', 'lohittamil',
        'likhan', 'samyakdevanagari', 'gurajada', 'notosansmonocjktc', 'syamalaramana', 'lohitmalayalam', 'notosansmonocjksc',
        'notosansmonocjkkr', 'notosansmonocjkhk', 'sreekrushnadevaraya', 'notosansmonocjkjp', 'kacsttitlel', 'navilu', 'kacstoffice',
        'ubuntucondensed', 'tibetanmachineuni', 'kacstletter', 'standardsymbolsps', 'ori1uni', 'raghumalayalamsans', 'aakar',
        'notomono', 'mukti', 'suranna', 'lohitkannada', 'dyuthi', 'meera', 'dhurjati', 'pothana2000', 'mandali', 'gubbi',
        'mallanna', 'gargi', 'notocoloremoji', 'samyakgujarati', 'chandas', 'kacstbook', 'kacstposter', 'padmaabold11', 'sahadeva',
        'kacstqurn', 'kacstnaskh', 'ntr', 'nakula', 'samanata', 'vemana2000', 'suruma', 'kacsttitle', 'samyakmalayalam']

        print(fonts)

    def working_fonts(self):
        # The fonts that even render
        fonts = [
        'tlwgtypo', 'dejavuserif', 'urwbookman', 'tlwgtypewriter', 'dejavusansmono', 'ubuntumono', 'rachana',
        'liberationmono', 'pottisreeramulu', 'anjalioldlipi', 'suravaram', 'notoserifcjksc', 'keraleeyam', 'c059',
        'garuda', 'nimbusmonops', 'notosansmono', 'notoserifcjktc', 'freesans', 'p052', 'liberationsansnarrow',
        'padaukbook', 'dejavusans', 'nimbussans', 'rasa', 'liberationsans', 'nimbussansnarrow', 'padmaa', 'notoserifcjkjp',
        'notoserifcjkhk', 'notoserifcjkkr', 'freeserif', 'abyssinicasil', 'uroob', 'yrsa', 'tlwgtypist', 'peddana',
        'freemono', 'gayathri', 'notosanscjkjp', 'notosanscjkhk', 'notosanscjkkr', 'loma', 'liberationserif', 'padauk',
        'ubuntu', 'notosanscjksc', 'laksaman', 'chilanka', 'notosanscjktc', 'kinnari', 'tlwgmono', 'ramaraja', 'mitra',
        'waree', 'sarai', 'manjari', 'umpush', 'z003', 'urwgothic', 'sawasdee', 'd050000l', 'timmana', 'norasi',
        'purisa', 'nimbusroman', 'khmeros', 'gidugu', 'lohitdevanagari', 'kalimati', 'khmerossystem', 'lohittelugu',
        'ramabhadra', 'nats', 'karumbi', 'phetsarathot', 'ani', 'tenaliramakrishna', 'jamrul', 'pagul', 'likhan',
        'gurajada', 'notosansmonocjktc', 'syamalaramana', 'notosansmonocjksc', 'notosansmonocjkkr', 'notosansmonocjkhk',
        'sreekrushnadevaraya', 'notosansmonocjkjp', 'ubuntucondensed', 'tibetanmachineuni', 'standardsymbolsps',
        'ori1uni', 'aakar', 'notomono', 'suranna', 'dyuthi', 'meera', 'dhurjati', 'pothana2000', 'mandali', 'mallanna',
        'gargi', 'chandas', 'padmaabold11', 'sahadeva', 'ntr', 'nakula', 'samanata', 'vemana2000', 'suruma', 'kacsttitle']

        print(fonts)

    def good_fonts(self):
        # Updated manualy as I find which fonts are better
        fonts = [
        'tlwgtypo', 'dejavuserif', 'urwbookman', 'tlwgtypewriter', 'dejavusansmono', 'ubuntumono', 'rachana',
        'liberationmono', 'pottisreeramulu', 'anjalioldlipi', 'suravaram', 'notoserifcjksc', 'keraleeyam', 'c059',
        'garuda', 'nimbusmonops', 'notosansmono', 'notoserifcjktc', 'freesans', 'p052', 'liberationsansnarrow',
        'padaukbook', 'dejavusans','nimbussans', 'rasa', 'liberationsans', 'nimbussansnarrow', 'padmaa', 'notoserifcjkjp',
        'notoserifcjkhk', 'notoserifcjkkr', 'freeserif', 'abyssinicasil', 'uroob', 'yrsa', 'tlwgtypist', 'peddana',
        'freemono', 'gayathri', 'notosanscjkjp', 'notosanscjkhk', 'notosanscjkkr', 'loma', 'liberationserif',
        'padauk', 'ubuntu', 'notosanscjksc', 'laksaman', 'chilanka', 'notosanscjktc', 'kinnari', 'tlwgmono', 'ramaraja',
        'waree', 'sarai', 'manjari', 'umpush', 'z003', 'urwgothic', 'sawasdee', 'timmana', 'norasi', 'purisa', 'nimbusroman',
        'khmeros', 'gidugu', 'lohitdevanagari', 'kalimati', 'khmerossystem', 'lohittelugu', 'ramabhadra', 'nats', 'karumbi',
        'phetsarathot', 'ani', 'tenaliramakrishna', 'jamrul', 'pagul', 'likhan', 'gurajada', 'notosansmonocjktc',
        'syamalaramana', 'notosansmonocjksc', 'notosansmonocjkkr', 'notosansmonocjkhk', 'sreekrushnadevaraya', 'notosansmonocjkjp',
        'ubuntucondensed', 'tibetanmachineuni', 'aakar', 'notomono', 'suranna', 'dyuthi', 'meera', 'dhurjati',
        'mandali', 'mallanna', 'gargi', 'chandas', 'padmaabold11', 'sahadeva', 'ntr', 'nakula', 'samanata', 'suruma']

        print(fonts)

    
