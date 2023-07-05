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
    for key, value in stats.items():
        print(key + ": " + format(value, '<.5g'))