import argparse
from pathlib import Path

from dpipe.io import choose_existing, save_json
from torch.utils.tensorboard import SummaryWriter

from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.players import MCTSZeroPlayer
from neuroxo.environment.field import Field
from neuroxo.algorithms.zero import run_train_val


def main():
    base_path = choose_existing(
        Path('/home/boris/Desktop/workspace/experiments/rl/zero/'),
        Path('/shared/experiments/rl/ttt_10x10/'),
        Path('/gpfs/gpfs0/b.shirokikh/experiments/rl/ttt_10x10/'),
    )

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
    parser.add_argument('--c_puct', required=False, type=float, default=5.0)
    parser.add_argument('--eps', required=False, type=float, default=0.25)
    parser.add_argument('--exploration_depth', required=False, type=int, default=8)
    parser.add_argument('--temperature', required=False, type=float, default=1.0)

    parser.add_argument('--reuse_tree', required=False, action='store_true', default=True)  # always True
    parser.add_argument('--deterministic_by_policy', required=False, action='store_true', default=False)
    parser.add_argument('--reduced_search', required=False, action='store_true', default=False)

    parser.add_argument('--lr_init', required=False, type=float, default=1e-4)
    parser.add_argument('--n_dec_steps', required=False, type=int, default=4)

    parser.add_argument('--n_epochs', required=False, type=int, default=200)
    parser.add_argument('--batch_size', required=False, type=int, default=128)
    parser.add_argument('--n_data_epochs_train', required=False, type=int, default=24)

    parser.add_argument('--augm', required=False, action='store_true', default=True)  # always True
    parser.add_argument('--shuffle_data', required=False, action='store_true', default=True)  # always True

    parser.add_argument('--n_val_games', required=False, type=int, default=60)
    parser.add_argument('--val_vs_random', required=False, action='store_true', default=False)

    args = parser.parse_known_args()[0]

    config = vars(args)
    config['in_channels'] = 2  # FIXME: should depend on feature generator
    # ### ###### ###

    # ### paths ###
    exp_path = base_path / args.exp_name
    exp_path.mkdir(exist_ok=True)

    logs_path = exp_path / 'logs'
    logs_path.mkdir(exist_ok=True)

    n_agents = len([*logs_path.glob('log*')])

    log_dir = logs_path / f'logs_{n_agents}'
    log_dir.mkdir(exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)

    configs_path = exp_path / 'configs'
    configs_path.mkdir(exist_ok=True)
    save_json(config, configs_path / f'config_train_{n_agents}.json')

    # ### ##### ###

    # ### experiment ###
    model = NeuroXOZeroNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)
    model_best = NeuroXOZeroNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)
    field = Field(args.n, k=args.k, field=None, device=args.device)

    player = MCTSZeroPlayer(model=model, field=field, device=args.device, n_search_iter=args.n_search_iter,
                            c_puct=args.c_puct, eps=args.eps, exploration_depth=args.exploration_depth,
                            temperature=args.temperature, reuse_tree=args.reuse_tree,
                            deterministic_by_policy=args.deterministic_by_policy, reduced_search=args.reduced_search)

    player_best = MCTSZeroPlayer(model=model_best, field=field, device=args.device, n_search_iter=args.n_search_iter,
                                 c_puct=args.c_puct, eps=args.eps, exploration_depth=args.exploration_depth,
                                 temperature=args.temperature, reuse_tree=args.reuse_tree,
                                 deterministic_by_policy=args.deterministic_by_policy,
                                 reduced_search=args.reduced_search)

    epoch2lr = {int(i * args.n_epochs / (args.n_dec_steps + 1)): args.lr_init * (0.5 ** i)
                for i in range(1, args.n_dec_steps + 1)}

    run_train_val(player, player_best, logger, exp_path, n_data_epochs_train=args.n_data_epochs_train,
                  n_epochs=args.n_epochs, n_val_games=args.n_val_games, batch_size=args.batch_size,
                  lr_init=args.lr_init, epoch2lr=epoch2lr, shuffle_data=args.shuffle_data, augm=args.augm,
                  val_vs_random=args.val_vs_random, preload_best=args.preload_best)


if __name__ == '__main__':
    main()
