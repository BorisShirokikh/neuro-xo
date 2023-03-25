import argparse
from pathlib import Path

from dpipe.io import choose_existing

from neuroxo.algorithms.zero import run_val


def main():

    parser = argparse.ArgumentParser()

    # ### CONFIG ###

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--preload_best', required=False, action='store_true', default=False)

    parser.add_argument('--n', required=False, type=int, default=10)
    parser.add_argument('--k', required=False, type=int, default=5)

    parser.add_argument('--n_blocks', required=False, type=int, default=11)
    parser.add_argument('--n_features', required=False, type=int, default=196)

    parser.add_argument('--n_search_iter', required=False, type=int, default=400)
    parser.add_argument('--n_search_iter_increment', required=False, type=int, default=20)
    parser.add_argument('--c_puct', required=False, type=float, default=5.0)
    parser.add_argument('--eps', required=False, type=float, default=0.25)
    parser.add_argument('--exploration_depth', required=False, type=int, default=8)
    parser.add_argument('--temperature', required=False, type=float, default=1.0)

    parser.add_argument('--reuse_tree', required=False, action='store_true', default=True)  # always True
    parser.add_argument('--deterministic_by_policy', required=False, action='store_true', default=False)
    parser.add_argument('--reduced_search', required=False, action='store_true', default=False)

    parser.add_argument('--n_val_games', required=False, type=int, default=128)  # 100 -> 0.60 wr; 200 -> 0.57 wr
    parser.add_argument('--n_jobs', required=False, type=int, default=1)

    parser.add_argument('--load_config_id', required=False, type=int, default=-1)

    args = parser.parse_known_args()[0]

    config = vars(args)
    config['in_channels'] = 2  # FIXME: should depend on feature generator

    # ### ###### ###

    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/'),
        Path('/shared/experiments/rl/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/'),
    )

    run_val(args, base_path)


if __name__ == '__main__':
    main()
