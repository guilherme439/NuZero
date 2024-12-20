import pytest

from Neural_Networks.Network_Manager import Network_Manager
from Neural_Networks.Architectures.ResNet import ResNet

from Games.SCS.SCS_Game import SCS_Game
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe


def test_ResNet_hex():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = ResNet(in_channels, policy_channels, num_filters=16, num_blocks=12, hex=True)
    print(model)

    network = Network_Manager(model)
    state = example_game.generate_state_image()
    policy, value = network.inference(state, False)

    print("Test completed successfully")

def test_ResNet():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = ResNet(in_channels, policy_channels, num_filters=16, num_blocks=12, hex=False)
    print(model)

    network = Network_Manager(model)
    state = example_game.generate_state_image()
    policy, value = network.inference(state, False)

    print("Test completed successfully")

