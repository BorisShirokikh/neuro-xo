import os
from pathlib import Path
import argparse

from torch.utils.tensorboard import SummaryWriter

from dpipe.io import choose_existing
from dpipe.torch import save_model_state, load_model_state

from ttt_lib.q_learning import PolicyNetworkQ, PolicyPlayer, train_q_learning, PolicyNetworkQ6, PolicyNetworkQ10, \
    PolicyNetworkQ10Light
from ttt_lib.field import Field


if __name__ == '__main__':

    Q_EXP_PATH = choose_existing(
        Path('/nmnt/x2-hdd/experiments/rl/'),
        Path('/experiments/borish/RL/ttt/mces')
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--n_step_q', required=True, type=int, choices=(1, 2))

    parser.add_argument('--field_size', required=False, type=int, default=3)
    parser.add_argument('--win_len', required=False, type=int, default=3)

    parser.add_argument('--light_model', required=False, action='store_true', default=False)

    parser.add_argument('--preload_path', required=False, type=str, default=None)

    args = parser.parse_known_args()[0]

    exp_path = Q_EXP_PATH / args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    log_dir = exp_path / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    n_step_q = args.n_step_q

    # hyperparameters:
    n = args.field_size
    kernel_len = args.win_len
    device = 'cuda'

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)

    cnn_features = (128, 64)
    if n == 3:
        model = PolicyNetworkQ(n=n, structure=cnn_features)
    elif n == 6:
        model = PolicyNetworkQ6(n=n, structure=cnn_features)
    elif n == 10:
        if args.light_model:
            model = PolicyNetworkQ10Light(n=n, structure=cnn_features)
        else:
            model = PolicyNetworkQ10(n=n, structure=cnn_features)
    else:
        raise ValueError('Only n = 3, 6, or 10 is supported:(')

    preload_path = args.preload_path
    if preload_path is not None:
        load_model_state(model, preload_path)

    n_episodes = int(1e6)
    eps_init = 0.5
    ep2eps = {int(1e5): 0.4, int(2e5): 0.35, int(3e5): 0.3, int(4e5): 0.25, int(5e5): 0.2,
              int(8e5): 0.15, int(9e5): 0.1}

    player = PolicyPlayer(model=model, field=field, eps=eps_init, device=device)

    # TRAIN:
    train_q_learning(player, logger, exp_path, n_episodes=n_episodes, n_step_q=n_step_q, ep2eps=ep2eps, n_duels=100)
    # SAVE:
    # save(sa2g, exp_path / 'sa2g.json')
    save_model_state(player.model, exp_path / 'model.pth')
