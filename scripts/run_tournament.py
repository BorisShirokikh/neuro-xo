import os
from itertools import product
from pathlib import Path
import argparse

from dpipe.io import choose_existing, save_json, load
import numpy as np
from dpipe.torch import load_model_state
from tqdm.auto import tqdm
from ttt_lib.self_games import play_duel

from ttt_lib.torch.module.policy_net import PolicyNetworkQ10
from ttt_lib.policy_player import PolicyPlayer
from ttt_lib.field import Field


if __name__ == '__main__':

    EXP_PATH = choose_existing(
        Path('/shared/experiments/rl/ttt_10x10/battle_room'),
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', required=False, type=str, default='cuda', choices=('cuda', 'cpu'))
    parser.add_argument('--n_games', required=False, type=int, default=50)
    parser.add_argument('--search_time', required=False, type=float, default=30)

    args = parser.parse_known_args()[0]

    log_path = EXP_PATH / 'results.json'
    models = [m for m in os.listdir(EXP_PATH) if m.endswith('.pth')]
    pairs = ['|'.join(pair) for pair in product(models, repeat=2) if pair[0] != pair[1]]

    if not log_path.exists():
        save_json(dict(), log_path, indent=0)

    log = load(log_path)

    for pair in tqdm(pairs):
        if pair in list(log.keys()):
            continue

        field = Field(n=10, kernel_len=5, field=None, device=args.device, check_device=args.device)

        model_x = PolicyNetworkQ10(n=10, structure=(128, 128))
        load_model_state(model_x, EXP_PATH / pair.split('|')[0])

        model_o = PolicyNetworkQ10(n=10, structure=(128, 128))
        load_model_state(model_o, EXP_PATH / pair.split('|')[1])

        player_x = PolicyPlayer(model=model_x, field=field, eps=None, device=args.device)
        player_o = PolicyPlayer(model=model_o, field=field, eps=None, device=args.device)

        scores = np.array([
            play_duel(player_x=player_x, player_o=player_o, return_result_only=True,
                      mcts_x=True, mcts_o=True, search_time_x=args.search_time, search_time_o=args.search_time)
            for _ in range(args.n_games)
        ])

        log[pair] = tuple([int(np.sum(scores == winner_i)) for winner_i in (1, 0, -1)])
        save_json(log, log_path, indent=0)

        print(pair.split('|')[0], ' (x) vs. ', pair.split('|')[1], ' (o):', log[pair], '(w-d-l)')
