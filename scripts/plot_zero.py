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
from neuroxo.utils import flush


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--vis_name', required=True, type=str)
    parser.add_argument('--eps', required=False, type=float, default=0.25)

    args = parser.parse_known_args()[0]

    # ### CONFIG ###
    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/zero/'),
        Path('/shared/experiments/rl/ttt_10x10/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/ttt_10x10/'),
    )
    # exp_name = 'zero_v2'
    exp_name = args.exp_name
    vis_name = args.vis_name

    device = 'cuda'

    n_blocks = 11
    n_features = 196

    n_search_iter = 1000
    c_puct = 5.
    eps = args.eps
    exploration_depth = 16
    temperature = 1.
    reuse_tree = True
    deterministic_by_policy = True
    # ### ###### ###

    os.makedirs(base_path / exp_name / vis_name, exist_ok=True)

    field = Field(n=10, kernel_len=5, device=device)
    model = NeuroXOZeroNN(n=10, in_channels=2, n_blocks=n_blocks, n_features=n_features)
    player = MCTSZeroPlayer(model=model, field=field, device=device, n_search_iter=n_search_iter, c_puct=c_puct,
                            eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                            reuse_tree=reuse_tree, deterministic_by_policy=deterministic_by_policy)
    load_model_state(player.model, base_path / exp_name / 'model.pth')

    t1 = time.perf_counter()
    s_history, _, a_history, o_history, _ = play_duel(player_x=player, player_o=player, self_game=True)
    t2 = time.perf_counter()

    flush('Performance time:', t2 - t1, 's')

    v_history, v_resign_history, proba_history = [], [], []
    for o in o_history:
        v_history.append(o[1])
        v_resign_history.append(o[2])
        proba_history.append(o[3])
    s, a, v, v_resign, p = np.array(s_history), np.array(a_history), np.array(v_history), np.array(v_resign_history),\
        np.array(proba_history)

    save(s, base_path / exp_name / vis_name / 's.npy')
    save(a, base_path / exp_name / vis_name / 'a.npy')
    save(v, base_path / exp_name / vis_name / 'v.npy')
    save(v_resign, base_path / exp_name / vis_name / 'v_resign.npy')
    save(p, base_path / exp_name / vis_name / 'p.npy')


if __name__ == '__main__':
    main()
