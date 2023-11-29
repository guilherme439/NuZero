
# NuZero

AlphaZero + DeepThinking + WarGames

Based on the [AlphaZero paper](https://www.semanticscholar.org/paper/A-general-reinforcement-learning-algorithm-that-and-Silver-Hubert/f9717d29840f4d8f1cc19d1b1e80c5d12ec40608)


## Features

* [x] Fully Assyncronous self-play, training and testing using [Ray](https://github.com/ray-project/ray).
* [x] Saves checkpoints during training and allows continuing previous training in case anything goes wrong.
* [x] Creates graphs for loss, win rates and others.
* [X] Allows defining any custom games, network arquitectures and agents.
* [x] Tic Tac Toe and SCS games already implemented.
* [X] Definition of any custom SCS games within the already implemented rules.
* [x] Creation of custom SCS markers for units.


### Features in development

* [] Better testing inteface 

### Future features

* [ ] Training and testing preset creation and saving
* [ ] Visual Interface for users to play SCS games
* [ ] Hyperparameter optimization
* [ ] Destributed training updates

### Current issues

* [ ] Bad GPU performance
* [ ] Others...

## Structure




## Getting started
### Installation

```bash
git clone https://github.com/guilherme439/NuZero
cd NuZero

pip install -r requirements.txt
```


### Interactive

To start using the system, you can choose to use interactive mode by running:
```bash
python Run.py --interactive
```
This will show you a series of prompts that will allow you to start training/testing by selecting from one of the available games, networks and configurations.

### Training

In order to start training with a specific configuration, the training presets should be used.
Training presets are defined inside ```Run.py ```. 

```bash
python Run.py --training-preset 0 
```

As an example, training preset 0 trains a recurrent network for tic tac toe, using an optimized configuration, while the remaining presets are defined for SCS games.


### Testing
To test a trained network just use/define a testing preset. Currently preset 0 tests and provides statistics for a pretrained tic tac toe model, while presets 1 and 2 are used to visualize or get statistics from SCS games.

```bash
python Run.py --testing-preset 0
```


## Configs
In order to train the networks, both Training and Search configurations are required. These are located in Configs/Config_files/Training/ or Search/ respectively.



## Related work

* [NotYet](www.google.com) (Not_an_author)


## Authors

* Guilherme Palma (guilherme.palma@tecnico.ulisboa.pt)

