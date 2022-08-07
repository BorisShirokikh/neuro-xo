import time
from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike, save, load
from dpipe.torch import to_var, save_model_state, load_model_state

from neuroxo.players import MCTSZeroPlayer, PolicyPlayer
from neuroxo.self_games import play_duel
from neuroxo.torch.model import optimizer_step
from neuroxo.torch.module.policy_net import RandomProbaPolicyNN
from neuroxo.utils import flush, update_lr, s2hhmmss


# ### single node training ###


def train_zero(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, logger: SummaryWriter, exp_path: PathLike,
               n_epochs: int = 100, n_episodes: int = 10000, n_val_games: int = 20, batch_size: int = 256,
               lr_init: float = 4e-3, epoch2lr: dict = None, augm: bool = True, shuffle_data: bool = True,
               best_model_name: str = 'model', winrate_th: float = 0.6, val_vs_random: bool = False):
    exp_path = Path(exp_path)
    best_model_path = exp_path / f'{best_model_name}.pth'
    if not best_model_path.exists():
        save_model_state(player_best.model, best_model_path)

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    logger_loss_step = 0
    for epoch in range(n_epochs):

        flush(f'>>> Generating dataset on epoch {epoch}:')
        f_epoch, pi_epoch, z_epoch = generate_train_dataset(player_best, n_episodes, augm=augm)

        flush(f'>>> Training NN on epoch {epoch}:')
        logger_loss_step = train(player, optimizer, logger, logger_loss_step, f_epoch, pi_epoch, z_epoch,
                                 batch_size=batch_size, shuffle=shuffle_data)

        flush(f'>>> Validating NN on epoch {epoch}:')
        winrate_vs_best = validate(player, player_best, logger, epoch, n_val_games, val_vs_random=val_vs_random)

        update_best_model(player, player_best, exp_path, epoch, winrate_vs_best, winrate_th, best_model_name)

        update_lr(optimizer, epoch2lr=epoch2lr, epoch=epoch)

        flush()


def generate_train_dataset(player: MCTSZeroPlayer, n_episodes_per_epoch: int = 10000, augm: bool = True):
    player.eval()
    f_epoch, pi_epoch, z_epoch = [], [], []
    for i in trange(n_episodes_per_epoch):
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
        f_history_augm, pi_history_augm = [], []
        for i in range(len(z_history)):
            f, pi = f_history[i], pi_history[i]
            if np.random.rand() <= 0.75:
                k = np.random.randint(low=1, high=4)
                f = np.rot90(f, k=k, axes=(-2, -1))
                pi = np.rot90(pi, k=k, axes=(-2, -1))
            if np.random.rand() <= 0.5:
                flip_axis = np.random.choice((-2, -1))
                f = np.flip(f, axis=flip_axis)
                pi = np.flip(pi, axis=flip_axis)
            f_history_augm.append(f)
            pi_history_augm.append(pi)
        f_history, pi_history = f_history_augm, pi_history_augm

    return f_history, pi_history, z_history


def train(player: MCTSZeroPlayer, optimizer: torch.optim.Optimizer, logger: SummaryWriter, prev_step: int,
          f_stack: list, pi_stack: list, z_stack: list,
          batch_size: int = 256, shuffle: bool = True, min_batch_size_fraction: float = 0.5):
    train_size = len(z_stack)

    if not isinstance(f_stack, np.ndarray):
        f_stack = np.concatenate(f_stack, axis=0, dtype=np.float32)
    if not isinstance(pi_stack, np.ndarray):
        pi_stack = np.concatenate(pi_stack, axis=0, dtype=np.float32)
    z_stack = np.float32(z_stack)

    if shuffle:
        idx = np.random.permutation(np.arange(train_size))
        f_stack, pi_stack, z_stack = f_stack[idx], pi_stack[idx], z_stack[idx]

    player.train()

    n_steps = train_size // batch_size + 1
    skip_last_batch = False

    for i in trange(n_steps):
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

    eps = player.eps
    eps_best = player_best.eps
    player.eps = None
    player_best.eps = None

    player.eval()
    player_best.eval()

    stats_xo = np.array([play_duel(player, player_best, return_result_only=True)
                         for _ in trange(n_val_games // 2)])
    stats_ox = np.array([play_duel(player_best, player, return_result_only=True)
                         for _ in trange(n_val_games // 2)])
    winrate_vs_best = np.mean(np.concatenate((stats_xo, -stats_ox)) == 1)
    logger.add_scalar('winrate/vs_best', winrate_vs_best, epoch)

    if val_vs_random:
        player_random = PolicyPlayer(model=RandomProbaPolicyNN(n=player.n, in_channels=player.model.in_channels),
                                     field=player.field, eps=1., device=player.device)
        player_random.eval()

        stats_xo = np.array([play_duel(player, player_random, return_result_only=True)
                             for _ in trange(n_val_games // 2)])
        stats_ox = np.array([play_duel(player_random, player, return_result_only=True)
                             for _ in trange(n_val_games // 2)])
        logger.add_scalar('winrate/vs_random', np.mean(np.concatenate((stats_xo, -stats_ox)) == 1), epoch)

        del player_random

    player.deterministic_by_policy = deterministic_by_policy
    player_best.deterministic_by_policy = deterministic_by_policy_best

    player.eps = eps
    player_best.eps = eps_best

    return winrate_vs_best


def update_best_model(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, exp_path: PathLike, epoch: int,
                      winrate_vs_best: float, winrate_th: float = 0.55, best_model_name: str = 'model'):
    if winrate_vs_best >= winrate_th:
        save_model_state(player_best.model, exp_path / f'{best_model_name}_{epoch}.pth')
        save_model_state(player.model, exp_path / f'{best_model_name}.pth')
        load_model_state(player_best.model, exp_path / f'{best_model_name}.pth')


# ### multiple nodes training ###


def run_data_generator(player_best: MCTSZeroPlayer, exp_path: PathLike,
                       models_folder: str = 'models', data_folder: str = 'data_epoch',
                       n_epochs: int = 100, n_episodes: int = 1000, augm: bool = True):
    exp_path = Path(exp_path)
    models_path = exp_path / models_folder
    models_path.mkdir(exist_ok=True)

    model_idx = init_or_load_best_model(player=player_best, models_path=models_path)

    for epoch in range(n_epochs):
        flush(f'>>> Generating dataset on epoch {epoch}:')

        data_path = exp_path / f'{data_folder}_{epoch}'
        data_path.mkdir(exist_ok=True)

        n_data_instances = len([*data_path.glob('*')])

        time_sum = 0

        while n_data_instances < n_episodes:
            time_start = time.perf_counter()

            episode_path = data_path / f'episode_{np.random.randint(1e18)}'
            if episode_path.exists():
                continue
            else:
                episode_path.mkdir(exist_ok=True)

            f_episode, pi_episode, z_episode = run_episode(player=player_best, augm=augm)

            save(np.int8(np.concatenate(f_episode, axis=0)), episode_path / 'f.npy')
            save(np.float32(np.concatenate(pi_episode, axis=0)), episode_path / 'pi.npy')
            save(np.int8(z_episode), episode_path / 'z.npy')

            n_data_instances = len([*data_path.glob('*')])

            time_finish = time.perf_counter()

            time_delta = time_finish - time_start
            time_sum += time_delta
            time_delta_est = time_sum / n_data_instances
            time_total_est = time_delta_est * n_episodes

            flush(f'>>> {n_data_instances}/{n_episodes} '
                  f'[{s2hhmmss(time_sum)}<{s2hhmmss(time_total_est - time_sum)},  {time_delta:.2f}s/it]')

            model_idx = init_or_load_best_model(player=player_best, models_path=models_path, model_idx=model_idx)

        flush()


def run_train_val(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, logger: SummaryWriter, exp_path: PathLike,
                  models_folder: str = 'models', data_folder: str = 'data_epoch', n_last_epochs_train: int = 10,
                  n_epochs: int = 100, n_episodes: int = 1000, n_val_games: int = 40, batch_size: int = 128,
                  lr_init: float = 4e-4, epoch2lr: dict = None, shuffle_data: bool = True,
                  winrate_th: float = 0.6, val_vs_random: bool = False):
    exp_path = Path(exp_path)
    models_path = exp_path / models_folder
    models_path.mkdir(exist_ok=True)

    init_or_load_best_model(player=player, models_path=models_path)
    init_or_load_best_model(player=player_best, models_path=models_path)

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    logger_loss_step = 0
    for epoch in range(n_epochs):
        flush(f'>>> Training on epoch {epoch}:')
        # TODO: loading dataset

        data_path = exp_path / f'{data_folder}_{epoch}'
        data_path.mkdir(exist_ok=True)

        n_data_instances = len([*data_path.glob('*')])

        while n_data_instances < n_episodes:
            flush(f'>>> >>> Found only {n_data_instances} / {n_episodes} training instances. Sleeping for 300s.')
            time.sleep(300)
            n_data_instances = len([*data_path.glob('*')])

        f_epoch, pi_epoch, z_epoch = load_train_dataset(exp_path, epoch, n_last_epochs_train, data_folder)

        flush(f'>>> Training NN on epoch {epoch}:')
        logger_loss_step = train(player, optimizer, logger, logger_loss_step, f_epoch, pi_epoch, z_epoch,
                                 batch_size=batch_size, shuffle=shuffle_data)

        flush(f'>>> Validating NN on epoch {epoch}:')
        winrate_vs_best = validate(player, player_best, logger, epoch, n_val_games, val_vs_random=val_vs_random)

        if winrate_vs_best >= winrate_th:
            save_model_state(player.model, models_path / f'model_{epoch + 1}.pth')
            load_model_state(player_best.model, models_path / f'model_{epoch + 1}.pth')

        update_lr(optimizer, epoch2lr=epoch2lr, epoch=epoch)

        flush()


def model_path2model_idx(p: PathLike):
    return int(str(p.name).strip('.pth').split('_')[-1])


def init_or_load_best_model(player: MCTSZeroPlayer, models_path: PathLike, model_idx: int = -1, init_idx: int = 0):
    model_paths = [*models_path.glob('model*.pth')]
    if not model_paths:
        save_model_state(player.model, models_path / f'model_{init_idx}.pth')
        return init_idx

    model_best_path = sorted(model_paths, key=model_path2model_idx)[-1]
    model_best_idx = model_path2model_idx(model_best_path)

    if model_idx < model_best_idx:
        load_model_state(player.model, model_best_path)
        return model_best_idx
    return model_idx


def load_train_dataset(exp_path: PathLike, epoch: int, n_last_epochs_train: int = 10, data_folder: str = 'data_epoch'):
    f_epoch, pi_epoch, z_epoch = [], [], []
    for e in range(max(0, epoch - n_last_epochs_train + 1), epoch + 1):
        data_path = exp_path / f'{data_folder}_{e}'
        for p in data_path.glob('episode_*'):
            try:
                f_epoch.append(load(p / 'f.npy'))
                pi_epoch.append(load(p / 'pi.npy'))
                z_epoch.append(load(p / 'z.npy'))
            except FileNotFoundError:  # for debug purposes: folder is created, but the data is not saved
                pass

    f_epoch = np.concatenate(f_epoch, axis=0)
    pi_epoch = np.concatenate(pi_epoch, axis=0)
    z_epoch = np.concatenate(z_epoch, axis=0)

    return np.float32(f_epoch), np.float32(pi_epoch), np.float32(z_epoch)
