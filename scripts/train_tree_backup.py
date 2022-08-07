import argparse
from pathlib import Path

from dpipe.io import choose_existing, save_json
from torch.utils.tensorboard import SummaryWriter

from neuroxo.torch.module.policy_net import ProbaPolicyNN
from neuroxo.players import PolicyPlayer
from neuroxo.environment.field import Field
from neuroxo.train_algorithms.tree_backup import train_tree_backup


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

    parser.add_argument('--n', required=False, type=int, default=10)
    parser.add_argument('--k', required=False, type=int, default=5)

    parser.add_argument('--n_blocks', required=False, type=int, default=11)
    parser.add_argument('--n_features', required=False, type=int, default=196)

    parser.add_argument('--n_step_backup', required=False, type=int, default=8)

    parser.add_argument('--eps', required=False, type=float, default=0.25)
    parser.add_argument('--mix_policies', required=False, action='store_true', default=False)

    parser.add_argument('--augm', required=False, action='store_true', default=True)  # always True

    parser.add_argument('--lr_init', required=False, type=float, default=4e-6)
    parser.add_argument('--n_dec_steps', required=False, type=int, default=100)

    parser.add_argument('--n_epochs', required=False, type=int, default=1000)
    parser.add_argument('--n_iters', required=False, type=int, default=1000)
    parser.add_argument('--min_batch_size', required=False, type=int, default=256)

    parser.add_argument('--n_val_games', required=False, type=int, default=400)
    parser.add_argument('--val_vs_random', required=False, action='store_true', default=True)  # always True

    parser.add_argument('--winrate_th', required=False, type=float, default=0.55)

    args = parser.parse_known_args()[0]

    config = vars(args)
    config['in_channels'] = 2  # FIXME: should depend on feature generator
    # ### ###### ###

    exp_path = base_path / args.exp_name
    exp_path.mkdir(exist_ok=True)

    save_json(config, exp_path / f'config_tb{args.n_step_backup}.json')

    log_dir = exp_path / 'logs'
    log_dir.mkdir(exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)

    model = ProbaPolicyNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)
    model_best = ProbaPolicyNN(args.n, in_channels=args.in_channels, n_blocks=args.n_blocks, n_features=args.n_features)

    field = Field(args.n, k=args.k, field=None, device=args.device)

    player = PolicyPlayer(model=model, field=field, device=args.device, eps=args.eps)
    player_best = PolicyPlayer(model=model_best, field=field, device=args.device, eps=args.eps)

    epoch2lr = {int(i * args.n_epochs / (args.n_dec_steps + 1)): args.lr_init * (0.5 ** i)
                for i in range(1, args.n_dec_steps + 1)}

    train_tree_backup(player, player_best, logger, exp_path, args.n_step_backup, mix_policies=args.mix_policies,
                      augm=args.augm, min_batch_size=args.min_batch_size, lr_init=args.lr_init, n_epochs=args.n_epochs,
                      n_iters=args.n_iters, n_val_games=args.n_val_games, val_vs_random=args.val_vs_random,
                      best_model_name='model', winrate_th=args.winrate_th, epoch2lr=epoch2lr)


if __name__ == '__main__':
    main()
