# RL for Tic-Tac-Toe 10x10

Code for the final project within the Reinforcement Learning 2021 course at Skoltech.

<div align="center">
<img src="https://user-images.githubusercontent.com/25771270/139085627-17feb7e2-e747-4b54-a927-384fe15af2f9.gif" width="250">
</div>

## Contribution
- [n-step off-Policy Tree Backup](neuroxo/train_algorithms/off_policy_tree_backup.py), [n-step Q(σ)-learinig](neuroxo/train_algorithms/q_learning.py) [[1]](#1)
- [Monte-Carlo Tree Search](https://github.com/BorisShirokikh/RL2021_Final_Project/blob/main/ttt_lib/monte_carlo_tree_search.py) [[2]](#2)


## Repository Structure
```
├── models                  # pre-trained agents to play with
│   └── ...
├── notebooks               # experiment results evaluation
│   └── ...
├── scripts
│   ├── q_learn_*x*.py      # dfferent field size model training
│   └── q_play_10x10.py     # manual playing against the pre-trained model
│
└── ttt_lib                 # project's library
    └── ...
```

## Installation
Execute from the directory you want the repo to be installed:

```
git clone https://github.com/BorisShirokikh/RL2021_Final_Project/
cd RL2021_Final_Project
pip install -e .
```


## References
<a id="1">[1]</a> Sutton R. S., Barto A. G. Reinforcement learning: An introduction. – MIT press, 2018.

<a id="2">[2]</a> Silver D. et al. Mastering the game of Go with deep neural networks and tree search //nature. – 2016. – Т. 529. – №. 7587. – С. 484-489.
