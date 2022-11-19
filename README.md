# RL for Tic-Tac-Toe 10x10

Code for the extended version of Tic-Tac-Toe game:
10x10 board, winner should place 5 "x" or "o" in a row/column/any diagonal.

We implement an RL agent trained via AlphaZero algorithm [[1]](#1).

<div align="center">
<img src="https://user-images.githubusercontent.com/9141666/202870740-6a9181f8-e1bb-48af-9d8b-2f1fb21c6997.gif" width="400">
</div>


## Repository Structure
```
├── agents                      # pre-trained agents to play with
│   └── ...
│
├── assets                      # images used in gui
│   └── ...
│
├── gui                         # classes to draw the game board
│   └── ...
│
├── neuroxo                     # project's library
│   └── ...
│
├── notebooks                   # some experimental visualization
│   └── ...
│
├── scripts
│   ├── run_zero_data_gen.py    # 1st part of training: continuously generate new data using the best model
│   └── run_zero_train_val.py   # 2nd part of training: trains the current model using generated data, runs validation against the best model
│
└── play.py                     # Our "main" function. Here, you can play against the RL agent or simply enjoy multiplayer.
```


## Installation
Execute from the directory you want the repo to be installed:

```
git clone git@github.com:BorisShirokikh/neuro-xo.git
cd neuro-xo
pip install -e .
```


## References
<a id="1">[1]</a> Silver, David, et al. "Mastering the game of go without human knowledge." nature 550.7676 (2017): 354-359.
