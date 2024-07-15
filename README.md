
# NuZero

AlphaZero + DeepThinking + WarGames

This system was developed as an attempt to tackle the enormous complexity of Wargames, more specifically Standard Combat Series (SCS) games, by combining the AlphaZero learning capabilities with the Deepthinking extrapolation capacity. The general idea was to train recurrent networks on small/simple maps using AlphaZero and then use the techniques described in the Deepthinking papers to extrapolate the learned strategies to the very large maps of SCS games. The system ended up developing into a wider project to accommodate a larger set of games, network architectures and configurations. More information about the system and its conceptualization can be found in my [Master Thesis](www.google.com)

Most important papers for this project:
* [Hexagdly](https://www.semanticscholar.org/paper/HexagDLy-Processing-hexagonally-sampled-data-with-Steppa-Holch/817d9ae8f6843d56ce984fa2eccb95ce97de4720?sort=is-influential)
* [AlphaZero](https://www.semanticscholar.org/paper/A-general-reinforcement-learning-algorithm-that-and-Silver-Hubert/f9717d29840f4d8f1cc19d1b1e80c5d12ec40608)
* [DT1](https://www.semanticscholar.org/paper/Can-You-Learn-an-Algorithm-Generalizing-from-Easy-Schwarzschild-Borgnia/941612bd6750efa76e1a75bdc64b6e3d7ed66457)
* [DT2](https://www.semanticscholar.org/paper/End-to-end-Algorithm-Synthesis-with-Recurrent-Bansal-Schwarzschild/c9143b978f91ee35429f1644a2266e5b036dad3a)


## Features

* [X] Options for both sequential and fully asynchronous self-play, training and testing using [Ray](https://github.com/ray-project/ray).
* [X] Large selection of hyperparameters that goes beyond the AlphaZero paper.
* [X] Allows defining custom games, network architectures and agents.
* [X] Already implemented network architectures for both hexagonal and orthogonal data.
* [X] Saves checkpoints during training so that training can be continued from any point.
* [X] Allows replay buffer saving/loading.
* [X] Creates graphs for loss, win rates and others.
* [X] Definition of any custom SCS games within the already implemented rules.
* [X] Simple visualization interface for SCS games.



## Getting started
### Installation

```bash
git clone https://github.com/guilherme439/NuZero
cd NuZero
```

You might want to create a virtual environment using:
```
python -m venv venv
```

or
```
virtualenv venv
```

Then activate it
```
source venv/bin/activate
```

Install the requirements:
```
pip install -r requirements.txt
```


### Training

In order to start training with a specific configuration, the training presets should be used.
Training presets are defined inside ```Run.py ```. 

```bash
python Run.py --training-preset 0 
```

As an example, training preset 0 trains a recurrent network for tic tac toe, using an optimized configuration, while the remaining presets are defined for SCS games.


### Testing
To test a trained network just use/define a testing preset. Currently, preset 0 tests and provides statistics for a pretrained tic tac toe model, while the remaing presets are used for SCS Games.

```bash
python Run.py --testing-preset 0
```

### Interactive

(currently not working in the latest version)

A command line interface is also available even though it does not support all the functionalities. The objective of this interface is giving new users a easy way to start using the system. To use it, simply run:
```bash
python Run.py --interactive
```
This will show you a series of prompts that will allow you to start training/testing by selecting from one of the available games, networks and configurations.

## Configs
In order to train the networks, both Training and Search configurations are required.
On the other hand, for testing-presets a Testing configuration is needed.
These configuration files are located in the Configs/ folder in their respective directories.


## Structure


The system can be ran in a variety of ways that can use different parts of the code, however these are the general responsibilities for each class:

Coordinators:
* AlphaZero - The main thread of the program. Responsible for training and plotting. Also launches the classes responsible for self-play.
* TestManager - Launches and manages the classes responsible for tests.

Workers: (usually several of these will run in parallel)

* Gamer - Plays individual self-play games.
* Tester - Runs individual test games.

Others:
* Explorer - Contains the methods necessary to run MCTS both in self-play and testing.

This diagram gives a generic view of the code logic:

(the diagram is not up to date, but still gives a general idea of the functioning of the system)
![ClassDiagram](Images/Classes_diagram.svg) 

When running sequentially, the AlphaZero instance will launch Gamers to play a certain number of games. When they finish playing those games, they terminate and AlphaZero executes a training step. At the end of this training step the Gamers are once again launched and the cycle continues. On the other hand, if running fully asynchronously, the Gamers are launched only once and they will keep playing games indefinitely and filling the replay buffer. In the meanwhile the AlphaZero instance will be training the network and storing it.

If the system is ran with asynchronous testing, the Test Manager will start in a separate process and the AlphaZero instance will check for which tests have concluded at the end of each training step. Otherwise, tests will run sequentially, meaning that control will switch to the Test Manager while running tests, and self-play and training only continue after the tests finish.

It is only possible to run sequential testing, if the system is not running in fully asynchronous mode.

## Notes
* This AlphaZero implementation was developed for two player games, despite also supporting single player games. In the AlphaZero version from the original paper, the state was always represented from the persective of the current player, which meant that the value function represented the advantage/disadvantage for the prespective of current player, this is, 1 was a victory for the current player, while -1 was a defeat. Since this implementation was developed with SCS games in mind, which display high objective variability and assymetry, we decided to use a static representation of the state, meaning that both players' units are represented the same way, indenpendetly of what player is currently playing. This ultimatly means that value function has a slightly different meaning, with 1 always representing a victory for player one, and -1 always representing a victory for player two. This change also affects the way in which the backpropagation is done at the end of each self-play game.

## Authors

* Guilherme Palma (guilherme.palma@tecnico.ulisboa.pt)

Publicly available alphazero pseudocode, as well as the deepthink github project were used as a base for some of the code and I also took ideas from other open-source AlphaGo/AlphaZero/MuZero implementations available on github.

