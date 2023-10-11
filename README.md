
# NuZero

AlphaZero + DeepThinking + WarGames


## Features

* [x] Self-play and Testing are done using [Ray](https://github.com/ray-project/ray) actors meaning it can be run multi-process/multi-computer.
* [x] Saves checkpoints during training and allows continuing previous training in case anything goes wrong.
* [x] Creates graphs for loss, win rates and others.
* [X] Allows defining any custom game or network arquitecture(using pytorch).
* [x] Tic Tac Toe and SCS games already implemented.
* [X] Definition of any custom SCS games within the already implemented rules.
* [x] Creation of custom SCS markers for units.

### Features in development

* [ ] Fully Assyncronous Training as described in the [AlphaZero paper](link_to_paper)
* [ ] CLI interactive mode for new users.
* [ ] 

### Future features

* [ ] Training and testing preset creation and saving
* [ ] Visual Interface for users to play SCS games

### Current issues

* [ ] Put the issues on github
* [ ] Bad GPU performance
* [ ] Others...

## Code structure




## Getting started
### Installation

```bash
git clone https://github.com/guilherme439/NuZero
cd NuZero

pip install -r requirements.txt
```

### Training

```bash
python Run.py --training-preset 0 
```

### Testing
```bash
python Run.py --testing-preset 1
```


### Configs



## Related work

* [NotYet](www.google.com) (Not_an_author)


## Authors

* Guilherme Palma (guilherme.palma@tecnico.ulisboa.pt)

