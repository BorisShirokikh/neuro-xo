import time
from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike, save, load
from dpipe.torch import to_var, save_model_state, load_model_state

from neuroxo.im.augm import get_dihedral_augm_numpy_fn
from neuroxo.players import MCTSZeroPlayer
from neuroxo.self_games import play_duel
from neuroxo.torch.model import optimizer_step
from neuroxo.utils import flush, update_lr, s2hhmmss, n2p_binomial_test


def run_episode(player: MCTSZeroPlayer, augm: bool = True):
    _, f_history, _, o_history, winner = play_duel(player, player, self_game=True)
    value = winner ** 2

    z_history = [value * (-1) ** i for i in range(len(f_history) + 1, 1, -1)]
    pi_history = [o[0][None, None] for o in o_history]

    if augm:
        for i in range(len(z_history)):
            augm_fn = get_dihedral_augm_numpy_fn()
            f_history[i] = augm_fn(f_history[i])
            pi_history[i] = augm_fn(pi_history[i])

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
             n_val_games: int = 100):
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

    player.deterministic_by_policy = deterministic_by_policy
    player_best.deterministic_by_policy = deterministic_by_policy_best

    player.eps = eps
    player_best.eps = eps_best

    return winrate_vs_best


def run_data_generator(player_best: MCTSZeroPlayer, exp_path: PathLike, models_folder: str = 'models',
                       data_folder: str = 'data', n_epochs: int = 100, n_episodes: int = 1000):
    exp_path = Path(exp_path)

    models_path = exp_path / models_folder
    models_path.mkdir(exist_ok=True)

    data_path = exp_path / data_folder
    data_path.mkdir(exist_ok=True)

    model_idx = init_or_load_best_model(player=player_best, models_path=models_path)

    for epoch in range(n_epochs):
        flush(f'>>> Generating dataset on epoch {epoch}:')

        data_epoch_path = data_path / f'data_epoch_{epoch}'
        data_epoch_path.mkdir(exist_ok=True)

        n_data_instances = len([*data_epoch_path.glob('*')])

        time_sum = 0

        while n_data_instances < n_episodes:
            time_start = time.perf_counter()

            episode_path = data_epoch_path / f'episode_{np.random.randint(1e18)}'
            if episode_path.exists():
                continue

            f_episode, pi_episode, z_episode = run_episode(player=player_best, augm=False)

            episode_path.mkdir(exist_ok=True)
            save(np.int8(np.concatenate(f_episode, axis=0)), episode_path / 'f.npy')
            save(np.float32(np.concatenate(pi_episode, axis=0)), episode_path / 'pi.npy')
            save(np.int8(z_episode), episode_path / 'z.npy')

            n_data_instances = len([*data_epoch_path.glob('*')])

            time_finish = time.perf_counter()

            time_delta = time_finish - time_start
            time_sum += time_delta
            time_delta_est = time_sum / n_data_instances
            time_total_est = time_delta_est * n_episodes

            flush(f'>>> {n_data_instances}/{n_episodes} '
                  f'[{s2hhmmss(time_sum)}<{s2hhmmss(time_total_est - time_sum)},  {time_delta_est:.2f}s/it]')

            model_idx = init_or_load_best_model(player=player_best, models_path=models_path, model_idx=model_idx)

        flush()


def run_train_val(player: MCTSZeroPlayer, player_best: MCTSZeroPlayer, logger: SummaryWriter, exp_path: PathLike,
                  models_folder: str = 'models', data_folder: str = 'data', n_data_epochs_train: int = 32,
                  n_epochs: int = 200, n_val_games: int = 100, batch_size: int = 512, lr_init: float = 1e-2,
                  epoch2lr: dict = None, augm: bool = True, shuffle_data: bool = True, preload_best: bool = True):
    exp_path = Path(exp_path)

    models_path = exp_path / models_folder
    models_path.mkdir(exist_ok=True)

    data_path = exp_path / data_folder
    data_path.mkdir(exist_ok=True)

    if preload_best:
        init_or_load_best_model(player=player, models_path=models_path)

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    winrate_th = n2p_binomial_test(n_val_games)

    logger_loss_step = 0
    for epoch in range(n_epochs):

        # 1. Loading and training:
        flush(f'>>> Training NN on epoch {epoch}:')
        data_epoch = None if (get_last_data_epoch(data_path) < (n_data_epochs_train + epoch)) else \
            (n_data_epochs_train + epoch)
        f_epoch, pi_epoch, z_epoch = load_train_dataset(data_path, n_data_epochs_train, augm, data_epoch=data_epoch)

        logger_loss_step = train(player, optimizer, logger, logger_loss_step, f_epoch, pi_epoch, z_epoch,
                                 batch_size=batch_size, shuffle=shuffle_data)

        # 3. Validating and updating:
        need_val = True
        while need_val:

            model_best_idx = init_or_load_best_model(player=player_best, models_path=models_path)
            flush(f'>>> Validating NN on epoch {epoch} against model_{model_best_idx}:')

            winrate_vs_best = validate(player, player_best, logger, epoch, n_val_games)

            if winrate_vs_best < winrate_th:
                need_val = False  # break
                flush(f'>>> The agent DOES NOT surpass model_{model_best_idx} '
                      f'with winrate {winrate_vs_best:.3f} (threshold is {winrate_th:.3f})')

            if get_model_best_idx(models_path) == model_best_idx:
                need_val = False  # break

                if winrate_vs_best >= winrate_th:
                    flush(f'>>> The agent DOES surpass model_{model_best_idx} '
                          f'with winrate {winrate_vs_best:.3f} (threshold is {winrate_th:.3f})')
                    save_model_state(player.model, models_path / f'model_{model_best_idx + 1}.pth')
                    load_model_state(player_best.model, models_path / f'model_{model_best_idx + 1}.pth')

        update_lr(optimizer, epoch2lr=epoch2lr, epoch=epoch)

        flush()


def get_model_best_idx(models_path: PathLike):
    return path2idx(sorted(models_path.glob('model*.pth'), key=path2idx)[-1])


def get_last_data_epoch(data_path: PathLike):
    return path2idx(sorted(data_path.glob('data_epoch_*'), key=path2idx)[-1])


def path2idx(p: PathLike):
    return int(str(p.name).strip('.pth').split('_')[-1])


def init_or_load_best_model(player: MCTSZeroPlayer, models_path: PathLike, model_idx: int = -1, init_idx: int = 0):
    model_paths = [*models_path.glob('model*.pth')]
    if not model_paths:
        save_model_state(player.model, models_path / f'model_{init_idx}.pth')
        return init_idx

    model_best_path = sorted(model_paths, key=path2idx)[-1]
    model_best_idx = path2idx(model_best_path)

    if model_idx < model_best_idx:
        load_model_state(player.model, model_best_path)
        return model_best_idx
    return model_idx


def load_train_dataset(data_path: PathLike, n_last_epochs_train: int = 8, augm: bool = True, data_epoch: int = None):
    data_path = Path(data_path)
    if data_epoch is None:
        data_epoch = get_last_data_epoch(data_path)

    f_epoch, pi_epoch, z_epoch = [], [], []
    for e in range(max(0, data_epoch - n_last_epochs_train + 1), data_epoch + 1):
        data_epoch_path = data_path / f'data_epoch_{e}'
        for p in data_epoch_path.glob('episode_*'):
            try:
                f, pi, z = load(p / 'f.npy'), load(p / 'pi.npy'), load(p / 'z.npy')
                if augm:
                    # TODO: we can create 8 instances of all augmentations instead of just one random
                    augm_fn = get_dihedral_augm_numpy_fn()
                    f, pi = augm_fn(f), augm_fn(pi)
                f_epoch.append(f)
                pi_epoch.append(pi)
                z_epoch.append(z)
            except FileNotFoundError:  # folder is created, but the data is not saved
                pass

    f_epoch = np.concatenate(f_epoch, axis=0)
    pi_epoch = np.concatenate(pi_epoch, axis=0)
    z_epoch = np.concatenate(z_epoch, axis=0)

    return np.float32(f_epoch), np.float32(pi_epoch), np.float32(z_epoch)
