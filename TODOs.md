# TO DO's:

[dificulty, priority]


### Configs
- Consider changing INI configs to YAML. [medium, low]

### Running the system
- Interactive mode [hard, medium]
- Creation of custom training/testing presets based on user interactive choices, probably using preset save files[hard, low]

### Explorer
- Use id() to get unique identifier when using state_dict (helps to reduce memory usage) [easy, low] {2}

### Implement Ray's features
- Use Ray's destributed training. [easy, low]
- Implement Ray's hyperparameter search [medium/hard, low]

### Training
- Serparar loops de treino e self-play para classes separadas para poder começar a fazer o continuous training. [hard, medium]
- Implement continuous training option (probably with no intermidiate testing). [hard, medium]

### Tester


### SCS Game
- Convert board to numpy array (allows indexing with tuples) [medium, low] {1}

### Graphs
- Allow spliting of more graphs other than policy

### Rendering and Iterface
- SCS marker creator interface [hard, low]
- Implement a user interface to play games vs trained AI [hard, low]

### Structure



<!---------------------------------------------------------------------------------------------------------------------------------------->

## Details

### Continuous training
- Meter Gamers a jogar 


<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues

1 - Might not be possible because of strange read-only bug when using np.arrays
2 - Might not be possible because of object lifetimes


<!---------------------------------------------------------------------------------------------------------------------------------------->


