'''
stats = \
{
"number_of_moves" : 0,
"average_children" : 0,
"final_tree_size" : 0,
"average_tree_size" : 0,
"final_bias_value" : 0,
"average_bias_value" : 0,
"average_value_score" : 0,
"average_prior_score" : 0,
}


'''
def print_stats(stats):
    print("\n")
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
    "average_value_score" : 0,
    "average_prior_score" : 0,
    }

    print("\n")
    size = len(stats_list)
    for stats in stats_list:
        for key, value in stats.items():
            tmp_stats[key] += value/size

    print_stats(tmp_stats)