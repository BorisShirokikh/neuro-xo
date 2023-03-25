import argparse
import time
from itertools import count
from pathlib import Path

import numpy as np
from dpipe.io import choose_existing, load, save
from dpipe.torch import load_model_state

from neuroxo.players import MCTSZeroPlayer
from neuroxo.environment import Field
from neuroxo.self_games import play_duel
from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.utils import path_mkdir, flush


def main():

    parser = argparse.ArgumentParser()

    # ### CONFIG ###
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--n', required=False, type=int, default=10)
    args = parser.parse_known_args()[0]
    # ### ###### ###

    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/'),
        Path('/shared/experiments/rl/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/'),
    )

    exp_path = base_path / f'ttt_{args.n}x{args.n}' / args.exp_name
    models_path = exp_path / 'models'
    configs_path = exp_path / 'configs'

    config = load([*configs_path.glob('config_train_*')][0])
    n = config['n']
    k = config['k']
    device = 'cuda'  # device = config['device']
    in_channels = config['in_channels']
    n_blocks = config['n_blocks']
    n_features = config['n_features']
    n_search_iter = 1600  # n_search_iter = config['n_search_iter']
    c_puct = config['c_puct']
    eps = 0  # eps = config['eps']
    exploration_depth = 0  # exploration_depth = config['exploration_depth']
    temperature = config['temperature']
    reuse_tree = config['reuse_tree']
    deterministic_by_policy = True  # deterministic_by_policy = config['deterministic_by_policy']
    reduced_search = False  # reduced_search = config['reduced_search']

    field = Field(n=n, k=k, field=None, device=device)

    model_0 = NeuroXOZeroNN(n, in_channels=in_channels, n_blocks=n_blocks, n_features=n_features)
    player_0 = MCTSZeroPlayer(model=model_0, field=field, device=device, n_search_iter=n_search_iter,
                              c_puct=c_puct, eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                              reuse_tree=reuse_tree,
                              deterministic_by_policy=deterministic_by_policy, reduced_search=reduced_search)

    model_1 = NeuroXOZeroNN(n, in_channels=in_channels, n_blocks=n_blocks, n_features=n_features)
    player_1 = MCTSZeroPlayer(model=model_1, field=field, device=device, n_search_iter=n_search_iter,
                              c_puct=c_puct, eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                              reuse_tree=reuse_tree,
                              deterministic_by_policy=deterministic_by_policy, reduced_search=reduced_search)

    games_path = path_mkdir(exp_path / 'games', exist_ok=True)

    best_model_idx = _get_model_best_idx(models_path)

    for game_idx in count():

        model_idx_0, model_idx_1 = np.random.randint(best_model_idx + 1, size=2)

        load_model_state(player_0.model, models_path / f'model_{model_idx_0}.pth')
        load_model_state(player_1.model, models_path / f'model_{model_idx_1}.pth')

        log = _log_game_to_disk(player_0, player_1, model_idx_0, model_idx_1, games_path)
        x, o, r, t, m = log['x'], log['o'], log['result'], log['time'], len(log['history'])
        flush(f">>> [iter {game_idx}] X ({x}) vs O ({o}): res={r}, time={t}, moves={m}.")

        log = _log_game_to_disk(player_0, player_1, model_idx_0, model_idx_1, games_path, change_players=True)
        x, o, r, t, m = log['x'], log['o'], log['result'], log['time'], len(log['history'])
        flush(f">>> [iter {game_idx}] X ({x}) vs O ({o}): res={r}, time={t}, moves={m}.")


def _log_game_to_disk(player_0, player_1, model_idx_0, model_idx_1, games_path, change_players: bool = False,
                      return_log: bool = True):
    player_x, player_o = (player_1, player_0) if change_players else (player_0, player_1)
    model_idx_x, model_idx_o = (model_idx_1, model_idx_0) if change_players else (model_idx_0, model_idx_1)

    t_start = time.perf_counter()
    _, _, a_history, _, result = play_duel(player_x, player_o)
    t_finish = time.perf_counter()

    log = {'x': f'model_{model_idx_x}', 'o': f'model_{model_idx_o}', 'result': result, 'history': a_history,
           'time': np.round(t_finish - t_start, decimals=1)}

    file = games_path / f'game_{np.random.randint(1e18)}.json'
    if not file.exists():
        save(log, file)

    if return_log:
        return log


def _get_model_best_idx(models_path):
    return _path2idx(sorted(models_path.glob('model*.pth'), key=_path2idx)[-1])


def _path2idx(p):
    return int(str(p.name).strip('.pth').split('_')[-1])


if __name__ == '__main__':
    main()
