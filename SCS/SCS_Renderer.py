import pygame
import time
import ray
import math

from enum import Enum

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
    
            self.render_board(screen, game)

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
    
    def render_board(self, screen, game):
        # For now it only renders square boards

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

    
