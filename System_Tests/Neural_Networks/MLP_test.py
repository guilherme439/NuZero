import pytest

from Neural_Networks.Network_Manager import Network_Manager
from Neural_Networks.Architectures.MLP_Network import MLP_Network as MLP_Net

from Games.SCS.SCS_Game import SCS_Game
from Games.Tic_Tac_Toe.tic_tac_toe import tic_tac_toe


def test_MLP():
    example_game = SCS_Game("Games/SCS/Game_configs/randomized_config_5.yml")
    
    x,y,z = example_game.get_action_space_shape()
    out_features = x*y*z
    model = MLP_Net(out_features)

    network = Network_Manager(model)

    policy, value = network.inference(example_game.generate_state_image(), False)

    print(f"\nPolicy:\n{policy}")
    print("\n\n\n")
    print(f"\nValue:\n{value}")
    print("Test completed successfully")

