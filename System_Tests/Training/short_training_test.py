import pytest

from Utils.Functions.general_utils import *
from Utils.Functions.loading_utlis import *
from Utils.Functions.ray_utils import *
from Utils.Functions.stats_utils import *
from Utils.Functions.yaml_utils import *

from Neural_Networks.Network_Manager import Network_Manager

from Neural_Networks.Architectures.MLP_Network import MLP_Network as MLP_Net
from Neural_Networks.Architectures.ConvNet import ConvNet
from Neural_Networks.Architectures.ResNet import ResNet
from Neural_Networks.Architectures.RecurrentNet import RecurrentNet

from Games.SCS.SCS_Game import SCS_Game
from Games.SCS.SCS_Renderer import SCS_Renderer
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe

from Training.AlphaZero import AlphaZero

def test_short_training():
    game_class = SCS_Game
    game_args_list = [["Games/SCS/Game_configs/randomized_config_5.yml"]]
    game = game_class(*game_args_list[0])

    train_config_path="Configs/Training/System_Tests/short_training_config.yaml"
    search_config_path="Configs/Search/System_Tests/short_search_config.yaml"

    state_set = None

    x,y,z = game.get_action_space_shape()
    out_features = x*y*z
    model = MLP_Net(out_features)

    print("\n")
    context = start_ray_local()
    alpha_zero = AlphaZero(game_class, game_args_list, train_config_path, search_config_path, model=model, state_set=state_set)
    alpha_zero.run()

    #TODO: Add more check conditions to make sure everything worked fine
    print("Test completed successfully")

