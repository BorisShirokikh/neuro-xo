from pathlib import Path
import argparse
import os
import time

import numpy as np
from dpipe.io import choose_existing, save
from dpipe.torch import load_model_state

from neuroxo.self_games import play_duel
from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.players import MCTSZeroPlayer
from neuroxo.environment.field import Field
from neuroxo.train_algorithms.zero import get_model_best_idx
from neuroxo.utils import flush


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', required=False, type=str, default='cuda')

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--vis_name', required=True, type=str)
    parser.add_argument('--deterministic', required=False, action='store_true', default=False)
    parser.add_argument('--full_search', required=False, action='store_true', default=False)

    args = parser.parse_known_args()[0]

    # ### CONFIG ###
    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/zero/'),
        Path('/shared/experiments/rl/ttt_10x10/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/ttt_10x10/'),
    )
    exp_name = args.exp_name
    vis_name = args.vis_name

    device = args.device

    n_blocks = 11
    n_features = 196

    reuse_tree = True

    n_search_iter = 1200
    c_puct = 5.
    temperature = 1.

    eps = 0. if args.deterministic else 0.25
    exploration_depth = 0 if args.deterministic else 12
    deterministic_by_policy = True if args.deterministic else False
    reduced_search = not args.full_search
    # ### ###### ###

    os.makedirs(base_path / exp_name / vis_name, exist_ok=True)

    field = Field(n=10, k=5, device=device)
    model = NeuroXOZeroNN(n=10, in_channels=2, n_blocks=n_blocks, n_features=n_features)
    player = MCTSZeroPlayer(model=model, field=field, device=device, n_search_iter=n_search_iter, c_puct=c_puct,
                            eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                            reuse_tree=reuse_tree, deterministic_by_policy=deterministic_by_policy,
                            reduced_search=reduced_search)

    models_path = base_path / exp_name / 'models'
    model_idx = get_model_best_idx(models_path=models_path)
    load_model_state(player.model, models_path / f'model_{model_idx}.pth')

    t1 = time.perf_counter()
    s_history, _, a_history, o_history, _ = play_duel(player_x=player, player_o=player, self_game=True)
    t2 = time.perf_counter()

    flush('Performance time:', t2 - t1, 's')

    pi_history, v_history, v_resign_history, proba_history, n_history = [], [], [], [], []
    for o in o_history:
        pi_history.append(o[0])
        v_history.append(o[1])
        v_resign_history.append(o[2])
        proba_history.append(o[3])
        n_history.append(o[4])

    save(np.array(s_history), base_path / exp_name / vis_name / 's.npy')
    save(np.array(a_history), base_path / exp_name / vis_name / 'a.npy')
    save(np.array(pi_history), base_path / exp_name / vis_name / 'pi.npy')
    save(np.array(v_history), base_path / exp_name / vis_name / 'v.npy')
    save(np.array(v_resign_history), base_path / exp_name / vis_name / 'v_resign.npy')
    save(np.array(proba_history), base_path / exp_name / vis_name / 'p.npy')
    save(np.array(n_history), base_path / exp_name / vis_name / 'n.npy')


if __name__ == '__main__':
    main()
