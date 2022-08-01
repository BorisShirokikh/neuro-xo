from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike
from dpipe.torch import to_var, save_model_state, load_model_state

from neuroxo.players import MCTSZeroPlayer, PolicyPlayer
from neuroxo.self_games import play_duel
from neuroxo.torch.model import optimizer_step
from neuroxo.torch.module.policy_net import RandomProbaPolicyNN


def train_zero(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, logger: SummaryWriter, exp_path: PathLike,
               n_epochs: int = 100, n_episodes_per_epoch: int = 10000, n_val_games: int = 400, batch_size: int = 256,
               lr_init: float = 4e-3, epoch2lr: dict = None, augm: bool = True, shuffle_data: bool = True,
               best_model_name: str = 'model', winrate_th: float = 0.55, val_vs_random: bool = False,):
    exp_path = Path(exp_path)
    best_model_path = exp_path / f'{best_model_name}.pth'
    if not best_model_path.exists():
        save_model_state(player_best.model, best_model_path)

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    logger_loss_step = 0
    for epoch in trange(n_epochs):

        f_epoch, pi_epoch, z_epoch = generate_train_dataset(player_best, n_episodes_per_epoch, augm=augm)

        logger_loss_step = train(player, optimizer, logger, logger_loss_step, f_epoch, pi_epoch, z_epoch,
                                 batch_size=batch_size, shuffle=shuffle_data)

        winrate_vs_best = validate(player, player_best, logger, epoch, n_val_games, val_vs_random=val_vs_random)

        update_best_model(player, player_best, exp_path, epoch, winrate_vs_best, winrate_th, best_model_name)

        update_lr(optimizer, epoch2lr=epoch2lr, epoch=epoch)


def generate_train_dataset(player: MCTSZeroPlayer, n_episodes_per_epoch: int = 10000, augm: bool = True):
    player.eval()
    f_epoch, pi_epoch, z_epoch = [], [], []
    for i in range(n_episodes_per_epoch):
        f_episode, pi_episode, z_episode = run_episode(player=player, augm=augm)
        f_epoch += f_episode
        pi_epoch += pi_episode
        z_epoch += z_episode
    return f_epoch, pi_epoch, z_epoch


def run_episode(player: MCTSZeroPlayer, augm: bool = True):
    _, f_history, _, o_history, winner = play_duel(player, player, self_game=True)
    value = winner ** 2

    z_history = [value * (-1) ** i for i in range(len(f_history) + 1, 1, -1)]
    pi_history = [o[0][None, None] for o in o_history]

    if augm:
        if np.random.rand() <= 0.75:
            k = np.random.randint(low=1, high=4)
            f_history = np.rot90(f_history, k=k, axes=(-2, -1))
            pi_history = np.rot90(pi_history, k=k, axes=(-2, -1))
        if np.random.rand() <= 0.5:
            flip_axis = np.random.choice((-2, -1))
            f_history = np.flip(f_history, axis=flip_axis)
            pi_history = np.flip(pi_history, axis=flip_axis)

        if isinstance(f_history, np.ndarray):
            f_history = f_history.tolist()
        if isinstance(pi_history, np.ndarray):
            pi_history = pi_history.tolist()

    return f_history, pi_history, z_history


def train(player: MCTSZeroPlayer, optimizer: torch.optim.Optimizer, logger: SummaryWriter, prev_step: int,
          f_stack: list, pi_stack: list, z_stack: list,
          batch_size: int = 256, shuffle: bool = True, min_batch_size_fraction: float = 0.5):
    train_size = len(z_stack)

    f_stack = np.concatenate(f_stack, axis=0, dtype=np.float32)
    pi_stack = np.concatenate(pi_stack, axis=0, dtype=np.float32)
    z_stack = np.float32(z_stack)

    if shuffle:
        idx = np.random.permutation(np.arange(train_size))
        f_stack, pi_stack, z_stack = f_stack[idx], pi_stack[idx], z_stack[idx]

    player.train()

    n_steps = train_size // batch_size + 1
    skip_last_batch = False

    for i in range(n_steps):
        z = to_var(z_stack[i * batch_size: (i + 1) * batch_size], device=player.device)
        if len(z) < batch_size * min_batch_size_fraction:
            skip_last_batch = True
            continue

        f = to_var(f_stack[i * batch_size: (i + 1) * batch_size], device=player.device)
        pi = to_var(pi_stack[i * batch_size: (i + 1) * batch_size], device=player.device)

        p, v = player.forward(f)

        loss = torch.mean((z - v) ** 2) - torch.mean(pi[p > 0] * torch.log(p[p > 0]))

        optimizer_step(optimizer=optimizer, loss=loss)
        logger.add_scalar('loss', loss.item(), prev_step + i)

    return prev_step + n_steps - int(skip_last_batch)


def validate(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, logger: SummaryWriter, epoch: int,
             n_val_games: int = 400, val_vs_random: bool = False):
    deterministic_by_policy = player.deterministic_by_policy
    deterministic_by_policy_best = player_best.deterministic_by_policy
    player.deterministic_by_policy = True
    player_best.deterministic_by_policy = True

    player.eval()
    player_best.eval()

    stats_xo = np.array([play_duel(player, player_best, return_result_only=True)
                         for _ in range(n_val_games // 2)])
    stats_ox = np.array([play_duel(player_best, player, return_result_only=True)
                         for _ in range(n_val_games // 2)])
    winrate_vs_best = np.mean(np.concatenate((stats_xo, -stats_ox)) == 1)
    logger.add_scalar('winrate/vs_best', winrate_vs_best, epoch)

    if val_vs_random:
        player_random = PolicyPlayer(model=RandomProbaPolicyNN(n=player.n, in_channels=player.model.in_channels),
                                     field=player.field, eps=1., device=player.device)
        player_random.eval()

        stats_xo = np.array([play_duel(player, player_random, return_result_only=True)
                             for _ in range(n_val_games // 2)])
        stats_ox = np.array([play_duel(player_random, player, return_result_only=True)
                             for _ in range(n_val_games // 2)])
        logger.add_scalar('winrate/vs_random', np.mean(np.concatenate((stats_xo, -stats_ox)) == 1), epoch)

        del player_random

    player.deterministic_by_policy = deterministic_by_policy
    player_best.deterministic_by_policy = deterministic_by_policy_best
    return winrate_vs_best


def update_best_model(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, exp_path: PathLike, epoch: int,
                      winrate_vs_best: float, winrate_th: float = 0.55, best_model_name: str = 'model'):
    if winrate_vs_best >= winrate_th:
        save_model_state(player_best.model, exp_path / f'{best_model_name}_{epoch}.pth')
        save_model_state(player.model, exp_path / f'{best_model_name}.pth')
        load_model_state(player_best.model, exp_path / f'{best_model_name}.pth')


def update_lr(optimizer: torch.optim.Optimizer, epoch2lr: dict, epoch: int):
    try:
        optimizer.param_groups[0]['lr'] = epoch2lr[epoch]
    except (KeyError, TypeError):  # (no such epochs in the dict, dict is None)
        pass
