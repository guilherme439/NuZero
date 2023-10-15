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


### Rendering and Iterface
- SCS marker creator interface
- Implement a user interface to play games vs trained AI [hard, low]

### Structure
- Create "Utility" folder for all the files with useful stuff like "stats_utilities.py" and "loss_functions.py"


<!---------------------------------------------------------------------------------------------------------------------------------------->

## Details

### Continuous training
- Mudar o codigo para cada gamer jogar até que o buffer tenha N jogos.
- Lançar X Gamers e um Trainer e treinar/jogar até chegar a um certo numero de training steps 


<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues

1 - Might not be possible because of strange read-only bug when using np.arrays
2 - Might not be possible because of object lifetimes


<!---------------------------------------------------------------------------------------------------------------------------------------->


