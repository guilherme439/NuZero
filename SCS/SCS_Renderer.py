import pygame
import time
import ray

import sys
sys.path.append("..")
from RemoteStorage import RemoteStorage

@ray.remote
class SCS_Renderer():

    def __init__(self, remote_storage):
        self.game_storage = remote_storage

    def render(self):

        pygame.init()
        
        # A remote game storage is used to update the game being displayed
        game = ray.get(self.game_storage.get_item.remote())

        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        ORANGE = (200, 100, 0)
        RED = (200, 0, 0)
        BROWN = (90, 50, 0)

        # Set the width and height of the output window, in pixels
        WINDOW_WIDTH = 1200
        WINDOW_HEIGHT = 900

        GAME_HEIGHT = game.getBoardHeight()
        GAME_WIDTH = game.getBoardWidth()
        
        # Set up the drawing window
        screen = pygame.display.set_mode([WINDOW_WIDTH, WINDOW_HEIGHT])

        time.sleep(0.6)
        # Run until user closes window
        running=True
        while running:

            game = ray.get(self.game_storage.get_item.remote())
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running=False

            # Fill the background with white
            screen.fill(WHITE)
    
            # Draw the board
            board_top_offset = 0.15*WINDOW_HEIGHT
            board_bottom_offset = 0.05*WINDOW_HEIGHT

            board_height = (WINDOW_HEIGHT - board_top_offset - board_bottom_offset)
            board_width = board_height

            tile_height = board_height/GAME_HEIGHT
            tile_width = tile_height

            tile_border_width = 2
            board_border_width = 8
            

            board_center = (WINDOW_WIDTH//2, board_top_offset + board_height/2)
            
            x_offset = board_center[0] - board_width//2
            y_offset = board_center[1] - board_height//2
            

            board_position = (x_offset-board_border_width, y_offset-board_border_width)
            board_dimensions = (board_width+(2*board_border_width), board_height+(2*board_border_width))
            board_border = pygame.Rect(board_position, board_dimensions)
            pygame.draw.rect(screen, BROWN, board_border, board_border_width)


            board = game.get_board()
            for i in range(GAME_HEIGHT):
                for j in range(GAME_WIDTH):

                    # x goes left and right
                    # j goes left and right
                    # y goes up and down
                    # i goes up and down

                    # TILES
                    x_position = ((tile_width)*j)+x_offset
                    y_position = ((tile_height)*i)+y_offset
                    tile_position = (x_position, y_position)
                    tile_dimensions = (tile_height, tile_width)
                    tile_rect = pygame.Rect(tile_position, tile_dimensions)
                    pygame.draw.rect(screen, BLACK, tile_rect, tile_border_width)

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


            text = "SCS BOARD!"
            if len(game.action_history) > 0:
                last_action = game.action_history[-1]
                text = "Turn: " + str(game.current_turn)
                #text = game.string_action(last_action)


            text_font = pygame.font.SysFont("any_font", 60)
            text_block = text_font.render(text, False, RED)
            text_rect = text_block.get_rect(center=(WINDOW_WIDTH/2, 50))
            screen.blit(text_block, text_rect)

            # Update de full display
            pygame.display.flip()

            # Limit fps
            time.sleep(0.6)
        
        # Done! Time to quit.
        pygame.quit()
        exit()