import argparse
from pathlib import Path

from dpipe.io import choose_existing

from neuroxo.algorithms.zero import run_sync


def main():

    parser = argparse.ArgumentParser()

    # ### CONFIG ###
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--n', required=False, type=int, default=10)
    parser.add_argument('--use_logs_id', required=False, type=int, default=None)
    args = parser.parse_known_args()[0]
    # ### ###### ###

    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/'),
        Path('/shared/experiments/rl/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/'),
    )

    run_sync(args, base_path)


if __name__ == '__main__':
    main()
