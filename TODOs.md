# TO DO's:

[dificulty, priority]


### Configs
- Consider changing configs to YAML. [medium, low]

### Explorer
- Use id() to get unique identifier when using state_dict (helps to reduce memory usage) [easy, low] (Might not be possible because of object lifetimes)

### Custom games
- Custom placement locations for reinforcements

### Implement Ray's features
- Use Ray's destributed training. [easy, low]
- Implement Ray's hyperparameter search [medium/hard, low]

### Continuous Training
- Serparar loops de treino e self-play para classes separadas para poder fazer o ponto seguinte. [hard, medium]
- Implement continuous training option (probably with no intermidiate testing). [hard, medium]

### Tester

### SCS Game
- Implement stacking [hard, high]
- Convert board to numpy array (allows indexing with tuples) [medium, medium]

### Graphs

### Rendering and Iterface
- Implement a user interface to play games vs trained AI

<!---------------------------------------------------------------------------------------------------------------------------------------->

## Details

### Continuous training
- Mudar o codigo para cada gamer jogar até que o buffer tenha N jogos.
- Lançar X Gamers e um Trainer e treinar/jogar até chegar a um certo numero de training steps 


<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues

- rede tem informaçao suficiente sobre a fase do jogo mesmo sem a representar

