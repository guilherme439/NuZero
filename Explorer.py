import math
import numpy as np

from scipy.special import softmax

from Node import Node

'''

    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀ 
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦ 
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃   
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀   
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀

'''

# The explorer runs searches. (explorer sounds better than searcher)
class Explorer():  

    def __init__(self, search_config, training, recurrent_iterations=1):
        
        self.config = search_config
        self.training = training
        self.recurrent_iterations = recurrent_iterations
        self.score_count = 0 # number of times score() was called, for stats calculation


    def run_mcts(self, network, game, subtree_root, state_dict=None):
        
        self.network = network
        search_start = subtree_root


        self.stats = \
        {
        "root_bias_value" : 0,
        "average_prior_score" : 0,
        "average_value_score" : 0,
        }

        if self.training:
            self.add_exploration_noise(search_start)
        
        num_searches = self.config.simulation["mcts_simulations"]
        for i in range(num_searches):
            node = search_start
            scratch_game = game.clone()
            search_path = [node]
            
            while node.expanded():
                action_i, node = self.select_child(node)
                action_coords = np.unravel_index(action_i, scratch_game.get_action_space_shape())
                scratch_game.step_function(action_coords)
                search_path.append(node)
            
            if node.is_terminal():
                value = node.value()
            else:
                value = self.evaluate(node, scratch_game, state_dict)
            
            self.backpropagate(search_path, value)

        root_bias = self.calculate_balancing_bias(search_start)
        self.stats["root_bias_value"] = root_bias
        self.stats["average_prior_score"] /= self.score_count
        self.stats["average_value_score"] /= self.score_count

        action = self.select_action(game, search_start)
        return action, search_start.children[action], self.stats
    

    def select_action(self, game, node):
        visit_counts = [(child.visit_count, action) for action, child in node.children.items()]

        if self.training:
            epsilon = np.random.random()
            if epsilon < self.config.exploration["epsilon_random_exploration"]:
                valid_actions_mask = game.possible_actions().flatten()
                n_valids = np.sum(valid_actions_mask)
                probs = valid_actions_mask/n_valids
                action_i = np.random.choice(game.get_num_actions(), p=probs)

            elif game.get_length() < self.config.exploration["number_of_softmax_moves"]:
                action_i = self.softmax_action(visit_counts)

            else:
                action_i = self.max_action(visit_counts)
        else:
            action_i = self.max_action(visit_counts)


        return action_i
    
    def select_child(self, node):
        _, action, child = max((self.score(node, child), action, child) for action, child in node.children.items())
        return action, child
    
    def calculate_balancing_bias(self, node):
        # Relative importance between value and prior as the game progresses
        pb_c_base = self.config.uct["pb_c_base"]
        pb_c_init = self.config.uct["pb_c_init"]

        return math.log((node.visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init
    
    def calculate_child_importance(self, parent, child):
        # Relative importance amongst children based on their visit counts
        return (math.sqrt(parent.visit_count) / (child.visit_count + 1))
    
    def score(self, parent, child):
        balance = self.calculate_balancing_bias(parent)
        child_factor = self.calculate_child_importance(parent, child)

        prior_score = child.prior * child_factor
        prior_score = prior_score * balance

        value_score = child.value()
        if parent.to_play == 2:
            value_score = (-value_score)
        # for player 2 negative values are good

        self.score_count += 1
        self.stats["average_prior_score"] += prior_score
        self.stats["average_value_score"] += value_score


        return prior_score + value_score

    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value	

    def evaluate(self, node, game, state_dict):
        node.to_play = game.get_current_player()

        # During testing we always use network predictions
        if self.training:
            use_terminal = self.config.simulation["use_terminal"]
        else:
            use_terminal = False

        if use_terminal and game.is_terminal():
            node.terminal_value = game.get_terminal_value()
            return node.terminal_value
        

        state = game.generate_state_image()
        if state_dict is not None:
            state_key = tuple(state.numpy().flatten())
            result = state_dict.get(state_key)
            
            if result:
                (action_probs, predicted_value) = result
            else:
                action_probs, predicted_value = self.network.inference(state, False, self.recurrent_iterations)
                state_dict[state_key] = (action_probs, predicted_value)
                

        else:
            action_probs, predicted_value = self.network.inference(state, False, self.recurrent_iterations)


        value = predicted_value.item()
        
        if not game.is_terminal():

            # Expand the node.
            valid_actions_mask = game.possible_actions().flatten()
            action_probs = action_probs.cpu()[0].numpy().flatten()
            
            
            probs = action_probs * valid_actions_mask # Use mask to get only valid moves
            total = np.sum(probs)


            for i in range(game.get_num_actions()):
                if valid_actions_mask[i]:
                    node.children[i] = Node(probs[i]/total)

        else:
            # This line will only be reached if use_terminal=False and the game is terminal
            node.terminal_value = game.get_terminal_value()
            # the node's terminal value is set so that the mcts knows this is a terminal node,
            # but it is not used during search, since the returned value is the one given by the neural network

        return value 

    def max_action(self, visit_counts):
        max_pair = max(visit_counts, key=lambda visit_action_pair: visit_action_pair[0])
        return max_pair[1]

    def softmax_action(self, visit_counts):
        counts = []
        actions = []
        for (count, action) in visit_counts:
            counts.append(count)	
            actions.append(action)

        final_counts = softmax(counts)
        #final_counts = counts/np.sum(counts)

        probs = np.asarray(final_counts, dtype=np.float64).astype('float64')
        probs /= np.sum(probs) # re-normalize to improve precison
        return np.random.choice(actions, p=probs)
    
    def add_exploration_noise(self, node):
        frac = self.config.exploration["root_exploration_fraction"]
        alpha = self.config.exploration["dist_alpha"]
        beta = self.config.exploration["dist_beta"]

        actions = node.children.keys()
        noise = np.random.gamma(alpha, beta, len(actions))
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def print_tree(self, root, action_space_shape):
        buffer = []
        print("\nRoot -> ")
        buffer.append((root, None, 0, None, 0))

        identifier = 0
        last_pid = 0    # pid -> parent id
        while len(buffer) > 0:
            identifier +=1
            (node, parent, p_id, action_coords, level) = buffer.pop(0)
            if last_pid != p_id:
                print("\n-\n")
            last_pid = p_id
            value_score = node.value()


            if parent:
                ucbScore = self.score(parent, node)
                print("Level: " + str(level) + " Parent_id: " + str(p_id) + " Node_id: " + format(identifier, '2') + 
                      " V: " + format(value_score, '.02') + " U: " + format(ucbScore, '.2') + " Visits: " + str(node.visit_count) +
                      " To_play: " + str(node.to_play) + " Terminal: " + str(node.terminal_value) + " NN_Prior: " + format(node.prior, '.02'))
            else:
                print("Node_id: " + str(identifier) + " Level: " + str(level) + " V: " + format(value_score, '.02') + " To_play: " + str(node.to_play))
            
            
            for (a, child) in node.children.items():
                if child.to_play!=-1:
                    act = np.unravel_index(a, action_space_shape)
                    buffer.append((child, node, identifier, act, level+1))		
        

        return
    
    def __str__():
        return "                                                                \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⣴⣶⣶⣿⣿⣿⣿⣿⣷⣶⣶⣦⣤⡀⠀⠀       \n \
    ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⢀⣤⠶⠶⠟⠛⠛⠛⠛⠻⠿⠷⣶⣦⣄⡀⠀⠀⠀⠀⠀         ⠀⠀⠀⠀⠀⠀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⣠⠞⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠻⢿⣷⣄⠀⠀         ⠀⠀⠀⠀⠀⠀⠰⠛⠋⠉⠉⠉⠀⠀⠀⠀⠀⠀⠉⠉⠉⠙⠛⠄⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⢠⡾⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣷⡀              ⠀⠀⢀⣀⣀⣀⣤⣤⣤⣤⣤⣤⣤⣤⣤⣤⣄⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⢠⡟⠀⠀⠀⠀⠀⢀⣠⣤⣴⠶⠶⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣄        ⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦ \n \
 ⠀⢀⣿⠁⠀⠀⠀⢀⣴⠟⠛⢿⣟⠛⢶⡀⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⡄⠀      ⠹⣷⡉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⡿⠛⠋⣡⣾⠃ \n \
 ⠀⢸⡏⠀⠀⠀⠀⣾⡇⠀⠀⠀⣿⠃⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠛⠇⠀      ⠀⠈⠻⣶⣀⣸⣿⡇⠀⢀⣭⣭⣭⠉⠉⠉⠉⣭⣭⣤⣤⡄⢸⣿⣇⣀⣶⠟⠁⠀ \n \
 ⠀⢸⡇⠀⠀⠀⠀⢹⣇⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⠀⢰⣶⣦⠀      ⠀⠀⠀⠈⣿⡏⠀⠀⠐⠉⢥⣤⠀⠀⠀⠀⠀⠀⣴⡤⠀⠀⠀⠀⢹⣿⠁⠀⠀⠀⠀\n \
 ⠀⠸⣧⠀⠀⠀⠀⠀⠻⣦⣀⠀⠀⠀⣀⣤⡾⠛⠉⠉⠉⠛⣷⡄⠀⠀⢈⡉⠋⠀      ⠀⠀⠀⠀⢹⣧⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡟⠀⠀⠀⠀⠀\n \
 ⠀⠀⢻⡆⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠉⠁⠀⠀⠀⠀⠀⠀⢸⣷⠀⠀⣿⡿⠀⠀      ⠀⠀⠀⠀⠀⠉⣿⡇⠀⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠀⢸⡿⠉⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠈⢻⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠇⠀⣸⣿⠃⠀⠀      ⠀⠀⠀⠀⠀⠀⠸⣷⡆⠀⠀⠶⢀⣀⣀⣀⣀⡀⠶⠀⠀⢰⣾⠇⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠻⣧⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⣴⣿⠃⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠘⢷⣟⠀⠀⠈⠉⠉⠉⠉⠁⠀⠀⣻⡾⠃⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠈⠛⢷⣤⣀⠀⠀⠀⠀⠀⠀⠀⢀⣠⡾⠋⠀⣠⣾⣿⠏⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣷⣟⠀⡀⠘⠃⢀⠀⣻⣾⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠛⠓⠶⠶⠶⠖⠛⠋⠁⠀⠀⣴⣿⣿⠃⠀⠀⠀⠀⠀      ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠿⣷⣤⣤⣾⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
 ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀\n \
        "
