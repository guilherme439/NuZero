import pytest

from Neural_Networks.Network_Manager import Network_Manager
from Neural_Networks.Architectures.RecurrentNet import RecurrentNet

from Games.SCS.SCS_Game import SCS_Game
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe


def test_RecurrentNet_hex():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    recurrent_iterations = 40
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = RecurrentNet(in_channels, policy_channels, num_filters=64, num_blocks=2, hex=True)

    print(model)

    network = Network_Manager(model)
    policy, value = network.inference(example_game.generate_state_image(), False, recurrent_iterations)

    print("Test completed successfully")

def test_RecurrentNet():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    recurrent_iterations = 40
    
    in_channels = example_game.get_state_shape()[0]
    policy_channels = example_game.get_action_space_shape()[0]
    model = RecurrentNet(in_channels, policy_channels, num_filters=64, num_blocks=2, hex=False)

    print(model)

    network = Network_Manager(model)
    policy, value = network.inference(example_game.generate_state_image(), False, recurrent_iterations)

    print("Test completed successfully")

