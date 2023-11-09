# TO DO's:

[dificulty, priority]


### Configs
- Consider changing INI configs to YAML. [medium, low]

### Running the system
- Creation of custom training/testing presets based on user interactive choices, probably using preset save files[hard, low]

### Explorer
- Use id() to get unique identifier when using state_dict (helps to reduce memory usage) [easy, low] {2}

### Implement Ray's features
- Use Ray's destributed training. [easy, low]
- Implement Ray's hyperparameter search [medium/hard, low]

### Training
- Fully implement continuous training option. [hard, medium]

### Tester
- Allow the tester to use state_dict per actor, and not just per game [medium, low]

### SCS Game
- Convert board to numpy array (allows indexing with tuples) [medium, low] {1}

### Graphs
- Allow spliting of more graphs other than policy

### Rendering and Iterface
- SCS marker creator interface [hard, low]
- Implement a user interface to play games vs trained AI [hard, low]

### Structure





<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues

1 - Might not be possible because of strange read-only bug when using np.arrays
2 - Might not be possible because of object lifetimes


<!---------------------------------------------------------------------------------------------------------------------------------------->


