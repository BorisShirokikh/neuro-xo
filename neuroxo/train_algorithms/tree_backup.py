from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike
from dpipe.torch import to_var, save_model_state, load_model_state

from neuroxo.environment.field import Field
from neuroxo.torch.model import optimizer_step
from neuroxo.im.augm import get_dihedral_augm_numpy_fn
from neuroxo.torch.module.policy_net import RandomProbaPolicyNN
from neuroxo.utils import flush, update_lr
from neuroxo.self_games import play_duel
from neuroxo.players import PolicyPlayer


def train_tree_backup(player: PolicyPlayer, player_best: PolicyPlayer, logger: SummaryWriter, exp_path: PathLike,
                      n_step_backup: int, mix_policies: bool = False, augm: bool = True, min_batch_size: int = 256,
                      lr_init: float = 4e-6, n_epochs: int = 1000, n_iters: int = 10000, n_val_games: int = 400,
                      val_vs_random: bool = True, best_model_name: str = 'model', winrate_th: float = 0.55,
                      epoch2lr: dict = None):
    exp_path = Path(exp_path)
    best_model_path = exp_path / f'{best_model_name}.pth'
    if not best_model_path.exists():
        save_model_state(player_best.model, best_model_path)

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    for epoch in range(n_epochs):

        flush(f'>>> Running epoch {epoch}:')

        for episode in trange(n_iters):

            # 1. Generating batch:
            f_history_stack, a_history_stack, q_history_stack, value_stack\
                = generate_training_batch(player, min_batch_size, mix_policies, augm)
            f_history, a_history, q_star_history = normalize_batch(f_history_stack, a_history_stack, q_history_stack,
                                                                   value_stack, n_step_backup, player.device)

            # 2. Training step:
            train_step(player, optimizer, logger, f_history, a_history, q_star_history, i=epoch * n_iters + episode)

        # 3. Validation
        winrate_vs_best = validate(player, player_best, logger, epoch, n_val_games, val_vs_random)

        # 4. Validation-based updates
        update_best_model(player, player_best, exp_path, epoch, winrate_vs_best, winrate_th, best_model_name)

        update_lr(optimizer, epoch2lr=epoch2lr, epoch=epoch)


def generate_training_batch(player: PolicyPlayer, min_batch_size: int = 256, mix_policies: bool = False,
                            augm: bool = True):
    player.eval()

    f_history_stack, a_history_stack, q_history_stack, value_stack = [], [], [], []
    while sum(map(len, q_history_stack)) < min_batch_size:
        f_history, a_history, q_history, value = play_custom_game(player, mix_policies, augm)
        # TODO: add games vs fixed opponent
        f_history_stack.append(f_history)
        a_history_stack.append(a_history)
        q_history_stack.append(q_history)
        value_stack.append(value)

    return f_history_stack, a_history_stack, q_history_stack, value_stack


def play_custom_game(player: PolicyPlayer, mix_policies: bool = False, augm: bool = True):
    field = Field(n=player.n, k=player.field.get_k(), field=None, device=player.device)
    player.set_field(field=field)

    eps = player.eps

    if mix_policies:
        if not eps:
            raise ValueError(f'Cannot use `mix policy` option if eps is None or 0, '
                             f'i.e., the agent uses on-policy or greedy strategies, respectively. '
                             f'Current eps is {eps}.')
        if np.random.random_sample() < eps:
            player.eps = None  # switch to on-policy

    f_history = []      # features
    a_history = []      # actions
    q_history = []      # policies

    v = field.get_value()  # value

    while v is None:

        if augm:
            augm_fn = get_dihedral_augm_numpy_fn()
            field.set_field(augm_fn(field.get_field()))

        f_history.append(field.get_features())

        a, _, q = player.action()

        a_history.append(a)
        q_history.append(q.max())

        v = field.get_value()

    player.eps = eps

    return f_history, a_history, q_history, v


def normalize_batch(f_history_stack, a_history_stack, q_history_stack, value_stack, n_step_backup: int = 8,
                    device: str = 'cuda'):
    f_history = [f for _f_history in f_history_stack for f in _f_history[::-1]]
    a_history = [a for _a_history in a_history_stack for a in _a_history[::-1]]

    inv = (-1) ** n_step_backup

    q_star_history = []
    for _q_history, v in zip(q_history_stack, value_stack):
        _q_history = _q_history[::-1]

        for t in range(len(_q_history)):
            q_star = v * (-1) ** t if (t < n_step_backup) else _q_history[t - n_step_backup] * inv
            q_star_history.append(q_star)

    f_history = to_var(np.concatenate(f_history, axis=0), device=device)
    q_star_history = to_var(q_star_history, device=device)

    return f_history, a_history, q_star_history


def train_step(player: PolicyPlayer, optimizer: torch.optim.Optimizer, logger: SummaryWriter,
               f_history, a_history, q_star_history, i):
    player.train()

    _, policy = player.forward(f_history)

    batch_size = len(a_history)
    q = policy.view(batch_size, -1)[(np.arange(batch_size), a_history)]

    loss = 0.5 * ((q - q_star_history) ** 2).mean()
    optimizer_step(optimizer=optimizer, loss=loss)
    logger.add_scalar('loss', loss.item(), i)


def validate(player: PolicyPlayer, player_best: PolicyPlayer, logger: SummaryWriter, epoch: int,
             n_val_games: int = 400, val_vs_random: bool = True):
    eps = player.eps
    eps_best = player_best.eps
    player.eps = None
    player_best.eps = None

    player.eval()
    player_best.eval()

    stats_xo = np.array([play_duel(player, player_best, return_result_only=True) for _ in range(n_val_games // 2)])
    stats_ox = np.array([play_duel(player_best, player, return_result_only=True) for _ in range(n_val_games // 2)])
    winrate_vs_best = np.mean(np.concatenate((stats_xo, -stats_ox)) == 1)
    logger.add_scalar('winrate/vs_best', winrate_vs_best, epoch)

    if val_vs_random:
        player_random = PolicyPlayer(model=RandomProbaPolicyNN(n=player.n, in_channels=player.model.in_channels),
                                     field=player.field, eps=1., device=player.device)
        player_random.eval()

        stats_xo = np.array([play_duel(player, player_random, True) for _ in range(n_val_games // 2)])
        stats_ox = np.array([play_duel(player_random, player, True) for _ in range(n_val_games // 2)])
        logger.add_scalar('winrate/vs_random', np.mean(np.concatenate((stats_xo, -stats_ox)) == 1), epoch)

        del player_random

    player.eps = eps
    player_best.eps = eps_best

    return winrate_vs_best


def update_best_model(player: PolicyPlayer, player_best: PolicyPlayer, exp_path: PathLike, epoch: int,
                      winrate_vs_best: float, winrate_th: float = 0.55, best_model_name: str = 'model'):
    if winrate_vs_best >= winrate_th:
        save_model_state(player_best.model, exp_path / f'{best_model_name}_{epoch}.pth')
        save_model_state(player.model, exp_path / f'{best_model_name}.pth')
        load_model_state(player_best.model, exp_path / f'{best_model_name}.pth')
