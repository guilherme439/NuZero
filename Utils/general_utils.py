import os   
import re 
import glob
import pickle
import torch

from Neural_Networks.Torch_NN import Torch_NN

from Utils.Caches.DictCache import DictCache
from Utils.Caches.KeylessCache import KeylessCache

def create_cache(cache_choice, max_size):
    if cache_choice == "dict":
        cache = DictCache(max_size)
    elif cache_choice == "keyless":
        cache = KeylessCache(max_size)
    elif cache_choice == "disabled":
        cache = None
    else:
        print("\nbad cache_choice")
        exit()
    return cache

def create_optimizer(model, optimizer_name, learning_rate, weight_decay=1.0e-7, momentum=0.9, nesterov=False):
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Bad optimizer config.\nUsing default optimizer (Adam)...")
    return optimizer


# ------------------------------------------------------ #
# ------------------  Loading/Saving  ------------------ #
# ------------------------------------------------------ #

def load_network_checkpoint(game_name, network_name, iteration_number):
    game_folder = "Games/" + game_name + "/"
    cp_network_folder = game_folder + "models/" + network_name + "/"
    if not os.path.exists(cp_network_folder):
        raise Exception("Could not find a model with that name.\nIf you are using Ray jobs with a working_directory,\nonly the models uploaded to git will be available.")
    
    buffer_path = cp_network_folder + "replay_buffer.cp"
    plot_path = cp_network_folder + "plot_data.pkl"

    if iteration_number == "auto":
        cp_paths = glob.glob(cp_network_folder + "*_cp")
        # finds all numbers in string -> gets the last one -> converts to int -> orders the numbers -> gets last number
        iteration_number = sorted(list(map(lambda str: int(re.findall('\d+',  str)[-1]), cp_paths)))[-1]    

    checkpoint_path =  cp_network_folder + network_name + "_" + str(iteration_number) + "_cp"
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_pickle_path =  cp_network_folder + "base_model.pkl"
    model = load_pickle(model_pickle_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer_pickle_path =  cp_network_folder + "base_optimizer.pkl"
    base_optimizer = load_pickle(optimizer_pickle_path)
    optimizer_dict = checkpoint["optimizer_state_dict"]

    scheduler_pickle_path =  cp_network_folder + "base_scheduler.pkl"
    base_scheduler = load_pickle(scheduler_pickle_path)
    scheduler_dict = checkpoint["scheduler_state_dict"]

    nn = Torch_NN(model)
    return nn, base_optimizer, optimizer_dict, base_scheduler, scheduler_dict, buffer_path, plot_path, iteration_number

def save_checkpoint(save_path, network, optimizer, scheduler):
    checkpoint = \
    {
    'model_state_dict': network.get_model().state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, save_path)

def load_yaml_config(yaml_parser, file_path):
    with open(file_path, 'r') as stream:
        config_dict = yaml_parser.load(stream)
    return config_dict

def save_yaml_config(yaml_parser, file_path, config_dict):  
    with open(file_path, 'w') as stream:
        yaml_parser.dump(config_dict, stream)

def load_pickle(pickle_path):
    with open(pickle_path, 'rb') as file:
        data = pickle.load(file)
    return data

def save_pickle(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)


# ------------------------------------------------------ #
# -----------------------  Stats  ---------------------- #
# ------------------------------------------------------ #
        
'''
stats = \
{
"number_of_moves" : 0,
"average_children" : 0,
"final_tree_size" : 0,
"average_tree_size" : 0,
"final_bias_value" : 0,
"average_bias_value" : 0,
}

'''

def print_stats(stats):
    print()
    for key, value in stats.items():
        print(key + ": " + format(value, '<.5g'))

def print_stats_list(stats_list):
    tmp_stats = {
    "number_of_moves" : 0,
    "average_children" : 0,
    "final_tree_size" : 0,
    "average_tree_size" : 0,
    "final_bias_value" : 0,
    "average_bias_value" : 0,
    }

    size = len(stats_list)
    for stats in stats_list:
        for key, value in stats.items():
            tmp_stats[key] += value/size

    print_stats(tmp_stats)