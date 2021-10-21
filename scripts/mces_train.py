import os
from pathlib import Path
import argparse

from tqdm import trange

from torch.utils.tensorboard import SummaryWriter

from dpipe.io import choose_existing
from dpipe.torch import save_model_state, load_model_state

from ttt_lib.monte_carlo_es import PolicyNetworkMCES, PolicyPlayer, train_on_policy_monte_carlo_es
from ttt_lib.field import Field

if __name__ == '__main__':

    MCES_EXP_PATH = choose_existing(
        Path('/nmnt/x2-hdd/experiments/rl'),
        Path('/experiments/borish/RL/ttt/mces')
    )

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', required=True, type=str)

    parser.add_argument('--preload_path', required=False, type=str, default=None)

    parser.add_argument('--field_size', required=False, type=int, default=10)
    parser.add_argument('--device', required=False, type=str, default='cuda')
    parser.add_argument('--n_episodes', required=False, type=int, default=100)
    parser.add_argument('--eps', required=False, type=float, default=0.1)
    parser.add_argument('--lr', required=False, type=float, default=0.001)
    parser.add_argument('--augm', required=False, action='store_true', default=False)

    args = parser.parse_known_args()[0]

    exp_path = MCES_EXP_PATH / args.exp_name
    os.makedirs(exp_path, exist_ok=True)

    log_dir = exp_path / 'logs'
    log_dir.mkdir()

    logger = SummaryWriter(log_dir=log_dir)

    preload_path = args.preload_path

    # hyperparameters:
    n = m = args.field_size
    device = args.device
    n_episodes = args.n_episodes
    eps = args.eps
    lr = args.lr
    augm = args.augm

    cnn_features = (64, 128)

    field = Field(n=n, m=m, device=device, check_device=device)

    model = PolicyNetworkMCES(n=n, m=m, structure=cnn_features)
    if preload_path is not None:
        load_model_state(model, Path(preload_path) / 'model.pth')

    player = PolicyPlayer(model=model, field=field, eps=eps, device=device)

    # TRAIN:
    sa2g = train_on_policy_monte_carlo_es(player, n_episodes=n_episodes, eps=eps, lr=lr,
                                          verbose_module=trange, augm=augm, logger=logger)
    # SAVE:
    # save(sa2g, exp_path / 'sa2g.json')
    save_model_state(player.model, exp_path / 'model.pth')
