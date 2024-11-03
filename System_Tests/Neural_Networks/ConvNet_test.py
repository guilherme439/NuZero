import pytest

from Neural_Networks.Network_Manager import Network_Manager

from Neural_Networks.Architectures.ConvNet import ConvNet

from Games.SCS.SCS_Game import SCS_Game
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe


def test_ConvNet_hex():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = ConvNet(in_channels, policy_channels, num_filters=32, num_layers=8, hex=True)
    print(model)

    network = Network_Manager(model)
    policy, value = network.inference(example_game.generate_state_image(), False)

    print("Test completed successfully")

def test_ConvNet():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = ConvNet(in_channels, policy_channels, num_filters=32, num_layers=8, hex=True)
    print(model)

    network = Network_Manager(model)
    policy, value = network.inference(example_game.generate_state_image(), False)

    print("Test completed successfully")

