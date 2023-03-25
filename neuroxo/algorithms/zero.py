import concurrent.futures
import time
from copy import deepcopy
from itertools import chain, count
from multiprocessing import cpu_count
from pathlib import Path
from subprocess import check_call

import numpy as np
from tqdm import trange
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike, save, load, save_json
from dpipe.torch import to_var, save_model_state, load_model_state, to_device

from neuroxo.environment import Field
from neuroxo.im.augm import get_dihedral_augm_numpy_fn
from neuroxo.players import MCTSZeroPlayer
from neuroxo.self_games import play_duel
from neuroxo.torch.model import optimizer_step
from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.utils import flush, n2p_binomial_test, s2hhmmss, path_mkdir


__all__ = ['run_data_generator', 'run_train', 'run_val', 'run_sync']
# TODO: split in 4 modules


# ######################################### Data Generation ##########################################################


def run_data_generator(args, base_path: PathLike):
    models_path, data_path, _, _, _ = _build_experiment(args, base_path, exp_type='data_gen')

    torch.multiprocessing.set_start_method('spawn')

    best_model_idx = _init_or_load_best_model(model=_get_player(args).model, models_path=models_path)
    torch.cuda.empty_cache()

    for epoch in count():
        flush(f'>>> Generating dataset on epoch {epoch}:')

        time_start = time.perf_counter()

        data_epoch_path = data_path / f'data_epoch_{epoch}'
        data_epoch_path.mkdir(exist_ok=True)

        n_jobs = _resolve_n_jobs(args.n_jobs)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(_run_and_save_episodes, args=args,
                                       models_path=models_path, data_epoch_path=data_epoch_path)
                       for _ in range(n_jobs)]
            _ = [future.result() for future in concurrent.futures.as_completed(futures)]

        time_finish = time.perf_counter()
        flush(f'>>> done in {s2hhmmss(time_finish - time_start)}.')

        # if best model has been successfully updated:
        new_best_model_idx = _get_model_best_idx(models_path)
        if new_best_model_idx > best_model_idx:
            args.n_search_iter += args.n_search_iter_increment
            best_model_idx = new_best_model_idx
            flush(f'>>> Best model has been updated => increasing search iterations to {args.n_search_iter}.')

        flush()


def _run_episode(player: MCTSZeroPlayer):
    _, f_history, _, o_history, winner = play_duel(player, player, self_game=True)
    value = winner ** 2

    z_history = [value * (-1) ** i for i in range(len(f_history) + 1, 1, -1)]
    pi_history = [o[0][None, None] for o in o_history]

    return f_history, pi_history, z_history


def _run_and_save_episodes(args, models_path: PathLike, data_epoch_path: PathLike):

    player_best = _get_player(args)
    model_idx = _init_or_load_best_model(model=player_best.model, models_path=models_path)

    n_data_instances = len([*data_epoch_path.glob('*')])

    while n_data_instances < args.n_episodes:
        episode_path = data_epoch_path / f'episode_{np.random.randint(1e18)}'
        if episode_path.exists():
            continue

        f_episode, pi_episode, z_episode = _run_episode(player=player_best)

        episode_path.mkdir(exist_ok=True)
        save(np.int8(np.concatenate(f_episode, axis=0)), episode_path / 'f.npy')
        save(np.float32(np.concatenate(pi_episode, axis=0)), episode_path / 'pi.npy')
        save(np.int8(z_episode), episode_path / 'z.npy')

        n_data_instances = len([*data_epoch_path.glob('*')])

        model_idx = _init_or_load_best_model(model=player_best.model, models_path=models_path, model_idx=model_idx)


# ########################################## Training Model ##########################################################


def run_train(args, base_path: PathLike,
              n_min_batches: int = 100, wait_for_data_gen_s: int = 600, data_exists: bool = False):
    models_path, data_path, logs_path, checkpoints_path, val_path = _build_experiment(args, base_path, exp_type='train')
    logger = SummaryWriter(log_dir=logs_path)

    player = _get_player(args)
    if args.preload_best:
        _init_or_load_best_model(model=player.model, models_path=models_path)
    optimizer = SGD(player.model.parameters(), lr=args.lr_init, momentum=0.9, weight_decay=1e-4, nesterov=True)

    train_step = 0
    checkpoint_idx = len([*checkpoints_path.glob('model*.pth')])
    best_model_idx = _get_model_best_idx(models_path)

    flush(f'>>> Training NN:')

    while True:

        f_epoch, pi_epoch, z_epoch = _load_train_dataset(data_path, args.n_data_epochs_train, args.augm, None,
                                                         batch_size=args.batch_size, n_min_batches=n_min_batches,
                                                         wait_for_data_gen_s=wait_for_data_gen_s,
                                                         data_exists=data_exists)
        data_exists = True  # waiting on the first epoch to generate enough data

        train_step = _train(player, optimizer, logger, train_step, f_epoch, pi_epoch, z_epoch,
                            batch_size=args.batch_size, data_dropout_rate=args.data_dropout_rate)

        new_best_model_idx = _get_model_best_idx(models_path)
        if new_best_model_idx > best_model_idx:
            optimizer.param_groups[0]['lr'] *= args.lr_dec_mult
            best_model_idx = new_best_model_idx
            flush(f'>>> Best model has been updated => setting learning rate to {optimizer.param_groups[0]["lr"]}.')

        if _all_checkpoints_are_validated(checkpoints_path, val_path):
            flush(f'>>> New checkpoint (model_{checkpoint_idx}.pth) is created at train step {train_step}.')

            player.eval()
            save_model_state(player.model, checkpoints_path / f'model_{checkpoint_idx}.pth')
            checkpoint_idx += 1


def _load_train_dataset(data_path: PathLike, n_data_epochs_train: int = 32, augm: bool = True, data_epoch: int = None,
                        batch_size: int = 512, n_min_batches: int = 100, wait_for_data_gen_s: int = 600,
                        data_exists: bool = False):
    def _load(_data_path, _n_data_epochs_train, _augm, _data_epoch):
        _data_path = Path(_data_path)
        if _data_epoch is None:
            try:
                _data_epoch = _get_last_data_epoch(_data_path)
            except IndexError:  # data_epoch_* folders have not been created yet
                _data_epoch = 0

        _f_epoch, _pi_epoch, _z_epoch = [], [], []
        for e in range(max(0, _data_epoch - _n_data_epochs_train + 1), _data_epoch + 1):
            data_epoch_path = _data_path / f'data_epoch_{e}'
            for p in data_epoch_path.glob('episode_*'):
                try:
                    f, pi, z = load(p / 'f.npy'), load(p / 'pi.npy'), load(p / 'z.npy')
                    if _augm:
                        augm_fn = get_dihedral_augm_numpy_fn()
                        f, pi = augm_fn(f), augm_fn(pi)
                    _f_epoch.append(f)
                    _pi_epoch.append(pi)
                    _z_epoch.append(z)
                except FileNotFoundError:  # folder is created, but the data is not saved
                    pass

        try:
            _f_epoch = np.concatenate(_f_epoch, axis=0)
            _pi_epoch = np.concatenate(_pi_epoch, axis=0)
            _z_epoch = np.concatenate(_z_epoch, axis=0)
        except ValueError:  # f, pi, and z are empty arrays [] (data has not been saved yet)
            pass

        return np.float32(_f_epoch), np.float32(_pi_epoch), np.float32(_z_epoch)

    if not data_exists:
        while _load(data_path, n_data_epochs_train, augm, data_epoch)[-1].size < batch_size * n_min_batches:
            flush(f'... waiting data for {wait_for_data_gen_s} s.')
            time.sleep(wait_for_data_gen_s)

    return _load(data_path, n_data_epochs_train, augm, data_epoch)


def _train(player: MCTSZeroPlayer, optimizer: torch.optim.Optimizer, logger: SummaryWriter, prev_step: int,
           f_stack: list, pi_stack: list, z_stack: list,
           batch_size: int = 256, data_dropout_rate: float = 0.2, min_batch_size_fraction: float = 0.5,
           verbose: bool = False):
    train_size = len(z_stack)

    if not isinstance(f_stack, np.ndarray):
        f_stack = np.concatenate(f_stack, axis=0, dtype=np.float32)
    if not isinstance(pi_stack, np.ndarray):
        pi_stack = np.concatenate(pi_stack, axis=0, dtype=np.float32)
    z_stack = np.float32(z_stack)

    idx = np.random.permutation(np.arange(train_size))[int(data_dropout_rate * train_size):]
    f_stack, pi_stack, z_stack = f_stack[idx], pi_stack[idx], z_stack[idx]

    player.train()

    n_steps = train_size // batch_size + 1
    skip_last_batch = False

    _range = trange if verbose else range
    for i in _range(n_steps):
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


def _all_checkpoints_are_validated(checkpoints_path: PathLike, val_path: PathLike):
    n_checkpoints_exist = len([*checkpoints_path.glob('model*.pth')])
    for i in range(n_checkpoints_exist):
        if not (val_path / f'val_{i}' / 'result.json').exists():
            return False
    return True


# ############################################## Validation ##########################################################


def run_val(args, base_path: PathLike, wait_for_checkpoint_s: float = 0.1):
    args = _init_or_load_val_config(args, base_path)
    models_path, _, _, checkpoints_path, val_path = _build_experiment(args, base_path, exp_type='val',
                                                                      save_config=(args.load_config_id < 0))

    torch.multiprocessing.set_start_method('spawn')

    _best_model_idx = _get_model_best_idx(models_path)
    flush(f'>>> Starting validation:\n')

    while True:

        checkpoint_idx = _find_invalidated_checkpoint(checkpoints_path, val_path, args.val_id, wait_for_checkpoint_s)
        player = _get_player(args)
        load_model_state(player.model, checkpoints_path / f'model_{checkpoint_idx}.pth')

        t_start = time.perf_counter()
        model_best_idx = _get_model_best_idx(models_path)
        flush(f'>>> Validating  checkpoint_{checkpoint_idx} against model_{model_best_idx}:')

        results_vs_best = _validate(args, model=player.model, models_path=models_path)

        new_model_best_idx = _get_model_best_idx(models_path)
        if new_model_best_idx != model_best_idx:
            flush(f'>>> (got new model): Validating  checkpoint_{checkpoint_idx} against model_{new_model_best_idx}:')
            results_vs_best = _validate(args, model=player.model, models_path=models_path)

        save_json({new_model_best_idx: results_vs_best},
                  val_path / f'val_{checkpoint_idx}' / f'results_{args.val_id}.json')

        t_finish = time.perf_counter()

        flush(f'>>> Validation took {s2hhmmss(t_finish - t_start)}.')

        torch.cuda.empty_cache()

        _new_model_best_idx = _get_model_best_idx(models_path)
        if _new_model_best_idx > _best_model_idx:
            args.n_search_iter += args.n_search_iter_increment
            _best_model_idx = _new_model_best_idx
            flush(f'>>> Best model has been updated => increasing search iterations to {args.n_search_iter}.')

        flush()


def _init_or_load_val_config(args, base_path, config_folder: str = 'configs'):
    if args.load_config_id >= 0:
        cfg = load(base_path / f'ttt_{args.n}x{args.n}' / args.exp_name / config_folder /
                   f'config_val_{args.load_config_id}.json')
        vars(args)['val_id'] = cfg['val_id']
    else:
        vars(args)['val_id'] = np.random.randint(1e18)
    return args


def _find_invalidated_checkpoint(checkpoints_path: PathLike, val_path: PathLike, val_id: int,
                                 wait_for_checkpoint_s: float = 0.1):

    while True:
        n_checkpoints_exist = len([*checkpoints_path.glob('model*.pth')])
        for i in range(n_checkpoints_exist):
            results_path = val_path / f'val_{i}' / f'results_{val_id}.json'
            if not results_path.exists():
                results_path.parent.mkdir(exist_ok=True)
                return i

        # flush(f'... waiting invalidated checkpoint for {wait_for_checkpoint_s} s.')
        time.sleep(wait_for_checkpoint_s)


def _validate(args, model: nn.Module, models_path: PathLike):

    n_jobs = _resolve_n_jobs(args.n_jobs)
    n_val_games_inst = args.n_val_games // n_jobs
    n_val_games_rest = args.n_val_games % n_jobs

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_validate_instance, args=args,
                                   models_path=models_path, n_val_games=n_val_games_inst, model=model)
                   for _ in range(n_jobs)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
        results_vs_best = list(chain(*results))

    results_vs_best_rest = _validate_instance(args, models_path=models_path, n_val_games=n_val_games_rest, model=model)
    results_vs_best += results_vs_best_rest.tolist()
    return results_vs_best


def _validate_instance(args, models_path: PathLike, n_val_games: int, model: nn.Module, verbose: bool = False):
    player, player_best = _get_player(args, return_paired_opponent=True, model_init_device='cpu')
    _init_or_load_best_model(player_best.model, models_path)
    to_device(player_best.model, device=args.device)
    player.model = deepcopy(model)

    player.deterministic_by_policy = True
    player_best.deterministic_by_policy = True

    player.eps = None
    player_best.eps = None

    player.eval()
    player_best.eval()

    _range = trange if verbose else range
    stats_xo = np.array([play_duel(player, player_best, return_result_only=True) for _ in _range(n_val_games // 2)])
    stats_ox = np.array([play_duel(player_best, player, return_result_only=True) for _ in _range(n_val_games // 2)])
    results_vs_best = np.concatenate((stats_xo, -stats_ox))

    return results_vs_best


# ################################################# Sync #############################################################


def run_sync(args, base_path: PathLike, config_folder: str = 'configs'):
    # TODO: clean old data
    models_path, data_path, logs_path, checkpoints_path, val_path = \
        _build_experiment(args, base_path, exp_type='train', save_config=False, use_logs_id=args.use_logs_id)
    logger = SummaryWriter(log_dir=logs_path)

    while True:
        model_best_idx = _get_model_best_idx(models_path)
        n_checkpoints_exist = len([*checkpoints_path.glob('model*.pth')])
        val_ids = [load(cfg_path)['val_id'] for cfg_path in
                   (Path(base_path) / f'ttt_{args.n}x{args.n}' / args.exp_name / config_folder).glob('config_val_*')]

        for checkpoint_idx in range(n_checkpoints_exist):
            val_i_path = val_path / f'val_{checkpoint_idx}'
            results_vs_best_paths = [val_i_path / f'results_{_id}.json' for _id in val_ids]

            if all([p.exists() for p in results_vs_best_paths]):
                result_path = val_i_path / 'result.json'
                if result_path.exists():
                    continue

                results_vs_best_dicts = list(map(load, results_vs_best_paths))
                best_model_idxs = list(map(lambda x: list(x)[0], results_vs_best_dicts))
                max_idx = max(best_model_idxs)

                if not np.all(np.array(best_model_idxs) == max_idx):
                    for idx, p in zip(best_model_idxs, results_vs_best_paths):
                        if idx != max_idx:
                            check_call(f'rm {str(p)}'.split(' '))

                else:
                    results_vs_best = np.concatenate(list(map(lambda x: list(x.values())[0], results_vs_best_dicts)))
                    winrate_vs_best = np.mean(results_vs_best == 1)
                    winrate_th = n2p_binomial_test(len(results_vs_best))

                    if winrate_vs_best < winrate_th:
                        flush(f'>>> The agent does NOT surpass model_{model_best_idx} '
                              f'with winrate {winrate_vs_best:.3f} (threshold is {winrate_th:.3f})')
                    else:
                        flush(f'>>> The agent DOES surpass model_{model_best_idx} '
                              f'with winrate {winrate_vs_best:.3f} (threshold is {winrate_th:.3f})')
                        source_model_path = checkpoints_path / f'model_{checkpoint_idx}.pth'
                        target_model_path = models_path / f'model_{model_best_idx + 1}.pth'
                        check_call(f'cp {source_model_path} {target_model_path}'.split(' '))

                    logger.add_scalar('winrate/vs_best', winrate_vs_best, checkpoint_idx)
                    save_json(winrate_vs_best, result_path)

            else:
                time.sleep(0.1)
                break


# ################################################# Utils ############################################################


def _get_model_best_idx(models_path: PathLike):
    return _path2idx(sorted(models_path.glob('model*.pth'), key=_path2idx)[-1])


def _get_last_data_epoch(data_path: PathLike):
    return _path2idx(sorted(data_path.glob('data_epoch_*'), key=_path2idx)[-1])


def _path2idx(p: PathLike):
    return int(str(p.name).strip('.pth').split('_')[-1])


def _init_or_load_best_model(model: nn.Module, models_path: PathLike, model_idx: int = -1, init_idx: int = 0):
    model_paths = [*models_path.glob('model*.pth')]
    if not model_paths:
        save_model_state(model, models_path / f'model_{init_idx}.pth')
        return init_idx

    model_best_path = sorted(model_paths, key=_path2idx)[-1]
    model_best_idx = _path2idx(model_best_path)

    if model_idx < model_best_idx:
        load_model_state(model, model_best_path)
        return model_best_idx
    return model_idx


def _resolve_n_jobs(n_jobs: int):
    n_cpu_avail = cpu_count()
    n_jobs = n_cpu_avail if (n_jobs == -1) else n_jobs
    n_jobs = 1 if (n_jobs == 0) else n_jobs
    return min(n_jobs, n_cpu_avail)


def _build_experiment(args, base_path: PathLike, exp_type: str, save_config: bool = True, use_logs_id: int = None,
                      models_folder: str = 'models', data_folder: str = 'data',
                      logs_folder: str = 'logs', config_folder: str = 'configs',
                      checkpoints_folder: str = 'checkpoints', val_folder: str = 'validation'):
    exp_path = path_mkdir(Path(base_path) / f'ttt_{args.n}x{args.n}' / args.exp_name, parents=True, exist_ok=True)

    models_path = path_mkdir(exp_path / models_folder, exist_ok=True)

    data_path = path_mkdir(exp_path / data_folder, exist_ok=True)

    configs_path = path_mkdir(exp_path / config_folder, exist_ok=True)
    n_configs = len([*configs_path.glob(f'config_{exp_type}_*')])

    config = vars(args)
    if save_config:
        save_json(config, configs_path / f'config_{exp_type}_{n_configs}.json')

    logs_path = path_mkdir(exp_path / logs_folder, exist_ok=True)
    if exp_type == 'train':
        if use_logs_id is None:
            logs_path = path_mkdir(logs_path / f'logs_{n_configs}', exist_ok=True)
        else:
            logs_path = logs_path / f'logs_{use_logs_id}'

    checkpoints_path = path_mkdir(exp_path / checkpoints_folder, exist_ok=True)

    val_path = path_mkdir(exp_path / val_folder, exist_ok=True)

    return models_path, data_path, logs_path, checkpoints_path, val_path


def _get_player(args, return_paired_opponent: bool = False, model_init_device: str = None):
    model_init_device = args.device if (model_init_device is None) else model_init_device

    field = Field(args.n, k=args.k, field=None, device=args.device)

    model = NeuroXOZeroNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)
    player = MCTSZeroPlayer(model=model, field=field, device=model_init_device, n_search_iter=args.n_search_iter,
                            c_puct=args.c_puct, eps=args.eps, exploration_depth=args.exploration_depth,
                            temperature=args.temperature, reuse_tree=args.reuse_tree,
                            deterministic_by_policy=args.deterministic_by_policy, reduced_search=args.reduced_search)
    player.device = args.device

    if return_paired_opponent:
        model_ = NeuroXOZeroNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)
        player_ = MCTSZeroPlayer(model=model_, field=field, device=model_init_device, n_search_iter=args.n_search_iter,
                                 c_puct=args.c_puct, eps=args.eps, exploration_depth=args.exploration_depth,
                                 temperature=args.temperature, reuse_tree=args.reuse_tree,
                                 deterministic_by_policy=args.deterministic_by_policy,
                                 reduced_search=args.reduced_search)
        player_.device = args.device
        return player, player_

    return player
