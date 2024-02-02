# TO DO's:

[dificulty, priority]


### Configs
- Consider changing INI configs to YAML. [medium, low]

### Running the system
- Creation of custom training/testing presets based on user interactive choices, probably using preset save files[hard, low]

### Explorer


### Implement Ray's features
- Use Ray's destributed training. [medium, low]
- Implement Ray's hyperparameter search [medium/hard, low]

### Training
- Save and load optimizer state when continuing training [medium, low]
- Add option to save replay buffer so that it can be loaded when continuing training (Needs to be implemented as an option since it will likely take a large space on disk)
- Change NN access logic, so that it can be better optimized for GPU. Use a single copy of the network that gets inference requests instead of multiple copies in each process. This can also help with cache performance since the cache can be centralized as well. [hard, medium]
- Implement PonderNet ?? [hard, low]

### Tester


### SCS Game
- Continue expanding game feature [medium, low]

### Graphs


### Rendering and Iterface
- SCS marker creator interface [hard, low]
- Implement a user interface to play games vs trained AI [hard, low]

### Structure





<!---------------------------------------------------------------------------------------------------------------------------------------->

## Issues





<!---------------------------------------------------------------------------------------------------------------------------------------->


