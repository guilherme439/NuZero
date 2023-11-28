# TO DO's:

[dificulty, priority]


### Configs
- Consider changing INI configs to YAML. [medium, low]

### Running the system
- Creation of custom training/testing presets based on user interactive choices, probably using preset save files[hard, low]

### Explorer
- Implement inference cache using cuckoo hashing (or something similar). The cache must not keep the keys to avoid high memory usage. [medium, high]

### Implement Ray's features
- Use Ray's destributed training. [easy, low]
- Implement Ray's hyperparameter search [medium/hard, low]

### Training


### Tester
- Implement policy, mcts and goal rush as Agents [high, high]
- Make test manager the responsible for all tests (including testing presets)[medium, high]

### SCS Game
- Convert board to numpy array (allows indexing with tuples) [medium, low] {1}

### Graphs


### Rendering and Iterface
- SCS marker creator interface [hard, low]
- Implement a user interface to play games vs trained AI [hard, low]
- Fix SCS_game's string representation for games with even number of rows [easy, medium]

### Structure





<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues

1 - Might not be possible because of strange read-only bug when using np.arrays
2 - Might not be possible because of object lifetimes


<!---------------------------------------------------------------------------------------------------------------------------------------->


