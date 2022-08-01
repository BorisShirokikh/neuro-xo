import os
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.players import MCTSZeroPlayer
from neuroxo.environment.field import Field
from neuroxo.train_algorithms.zero import train_zero


if __name__ == '__main__':
    # ### CONFIG ###
    base_path = Path('/home/boris/Desktop/workspace/experiments/rl/zero/')
    exp_name = 'test_zero'

    device = 'cpu'

    n = 10
    kernel_len = 5

    in_channels = 2  # FIXME: should depend on feature generator
    n_blocks = 2
    n_features = 16

    n_search_iter = 100
    c_puct = 5.
    eps = 0.25
    exploration_depth = 16
    temperature = 1.
    reuse_tree = True
    deterministic_by_policy = False

    lr_init = 3e-4
    n_dec_steps = 4

    n_epochs = 3
    n_episodes_per_epoch = 10
    n_val_games = 10
    batch_size = 4

    augm = True
    shuffle_data = True

    winrate_th = 0.55

    val_vs_random = True
    # ### ###### ###

    exp_path = base_path / exp_name
    os.makedirs(exp_path, exist_ok=True)

    log_dir = exp_path / 'logs'
    log_dir.mkdir(exist_ok=True)

    logger = SummaryWriter(log_dir=log_dir)

    field = Field(n=n, kernel_len=kernel_len, device=device)

    model = NeuroXOZeroNN(n=n, in_channels=in_channels, n_blocks=n_blocks, n_features=n_features)
    model_best = NeuroXOZeroNN(n=n, in_channels=in_channels, n_blocks=n_blocks, n_features=n_features)

    player = MCTSZeroPlayer(model=model, field=field, device=device, n_search_iter=n_search_iter, c_puct=c_puct,
                            eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                            reuse_tree=reuse_tree, deterministic_by_policy=deterministic_by_policy)

    player_best = MCTSZeroPlayer(model=model_best, field=field, device=device, n_search_iter=n_search_iter,
                                 c_puct=c_puct, eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                                 reuse_tree=reuse_tree, deterministic_by_policy=deterministic_by_policy)

    epoch2lr = {int(i * n_epochs / (n_dec_steps + 1)): lr_init * (0.5 ** i) for i in range(1, n_dec_steps + 1)}

    train_zero(player, player_best, logger, exp_path, n_epochs=n_epochs, n_episodes_per_epoch=n_episodes_per_epoch,
               n_val_games=n_val_games, batch_size=batch_size, lr_init=lr_init, epoch2lr=epoch2lr, augm=augm,
               shuffle_data=shuffle_data, best_model_name='model', winrate_th=winrate_th, val_vs_random=val_vs_random)
