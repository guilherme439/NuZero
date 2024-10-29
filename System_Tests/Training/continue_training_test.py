import pytest

from Utils.Functions.ray_utils import *

from Neural_Networks.Architectures.MLP_Network import MLP_Network as MLP_Net

from Games.SCS.SCS_Game import SCS_Game
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Training.AlphaZero import AlphaZero

def test_continue_training():
    game_class = SCS_Game
    game_args_list = [["Games/SCS/Game_configs/randomized_config_5.yml"]]

    train_config_path="Configs/Training/System_Tests/continue_training_config.yaml"
    search_config_path="Configs/Search/System_Tests/continue_search_config.yaml"

    state_set = None

    print("\n")
    context = start_ray_local()
    alpha_zero = AlphaZero(game_class, game_args_list, train_config_path, search_config_path)
    alpha_zero.run()

    #TODO: Add more check conditions to make sure everything worked fine
    print("Test completed successfully")

