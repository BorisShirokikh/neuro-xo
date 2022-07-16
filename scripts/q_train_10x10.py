import os
from pathlib import Path
import argparse

from torch.utils.tensorboard import SummaryWriter
from dpipe.io import choose_existing
from dpipe.torch import load_model_state

from ttt_lib.torch.module.policy_net import PolicyNetworkQ10Light, PolicyNetworkQ10
from ttt_lib.policy_player import PolicyPlayer
from ttt_lib.train_algorithms.off_policy_tree_backup import train_tree_backup
from ttt_lib.field import Field


if __name__ == '__main__':

    Q_EXP_PATH = choose_existing(
        Path('/nmnt/x4-hdd/experiments/rl/q_10x10/'),
        Path('/home/boris/Desktop/workspace/experiments/rl/q_10x10/'),
        # Path('/shared/experiments/rl/q_10x10/')
        Path('/shared/experiments/rl/ttt_10x10/')
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--device', required=False, type=str, default='cuda', choices=('cuda', 'cpu'))

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--method', required=False, type=str, default='TB', choices=('TB', 'REINFORCE'))
    parser.add_argument('--n_episodes', required=False, type=int, default=2000000)
    parser.add_argument('--n_step_q', required=False, type=int, default=8)
    parser.add_argument('--episodes_per_epoch', required=False, type=int, default=10000)

    parser.add_argument('--n_val_games', required=False, type=int, default=400)

    parser.add_argument('--random_starts', required=False, action='store_true', default=False)
    parser.add_argument('--random_starts_max_depth', required=False, type=int, default=10)

    parser.add_argument('--lr_init', required=False, type=float, default=4e-6)
    parser.add_argument('--eps_init', required=False, type=float, default=0.2)

    parser.add_argument('--preload_path', required=False, type=str, default=None)
    parser.add_argument('--large_model', required=False, action='store_true', default=False)

    args = parser.parse_known_args()[0]

    exp_path = Q_EXP_PATH / args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    log_dir = exp_path / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    # hyperparameters:
    n = 10
    kernel_len = 5

    n_episodes = args.n_episodes
    episodes_per_epoch = args.episodes_per_epoch
    n_epochs = n_episodes // episodes_per_epoch

    n_dec_steps = 4
    lr_init = args.lr_init
    epoch2lr = {int(i * n_epochs / (n_dec_steps + 1)): lr_init * (0.5 ** i) for i in range(1, n_dec_steps + 1)}

    eps_init = 0.4
    epoch2eps = {
        **{int(i * n_epochs / (n_dec_steps + 1)): eps_init * (n_dec_steps - i) / n_dec_steps
           for i in range(1, n_dec_steps)}, int(n_dec_steps * n_epochs / (n_dec_steps + 1)): None

    }

    device = args.device

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)

    cnn_features = (128, 64)
    if args.large_model:
        model = PolicyNetworkQ10(n=n, structure=cnn_features)
        model_opp = PolicyNetworkQ10(n=n, structure=cnn_features)
    else:
        model = PolicyNetworkQ10Light(n=n, structure=cnn_features)
        model_opp = PolicyNetworkQ10Light(n=n, structure=cnn_features)

    if args.preload_path is not None:
        load_model_state(model, args.preload_path)

    player = PolicyPlayer(model=model, field=field, eps=eps_init, device=device)
    opponent = PolicyPlayer(model=model_opp, field=field, eps=eps_init, device=device)

    # TRAIN:
    train_tree_backup(player=player, opponent=opponent, logger=logger, models_bank_path=exp_path, method=args.method,
                      n_episodes=n_episodes, n_step_q=args.n_step_q, episodes_per_epoch=episodes_per_epoch,
                      n_val_games=args.n_val_games,
                      random_starts=args.random_starts, random_starts_max_depth=args.random_starts_max_depth,
                      lr_init=lr_init, epoch2lr=epoch2lr, epoch2eps=epoch2eps,
                      best_model_name='model.pth', winrate_th=0.55)
