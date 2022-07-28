from copy import deepcopy

import numpy as np
from dpipe.io import PathLike
from dpipe.torch import load_model_state
from torch.utils.tensorboard import SummaryWriter

from neuroxo.environment.field import Field
from neuroxo.policy_player import PolicyPlayer
from neuroxo.self_games import play_duel
from neuroxo.torch.module.policy_net import PolicyNetworkRandom


def validate(epoch: int, player: PolicyPlayer, logger: SummaryWriter, n: int, n_games: int = 400,
             opponent_model_path: PathLike = None, return_winrate_vs_opponent: bool = False):
    device = player.device

    validation_field = Field(n=n, kernel_len=player.field.kernel_len, device=device, check_device=device)

    player_model_0 = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)
    player_model_1 = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)

    player_random = PolicyPlayer(model=PolicyNetworkRandom(n=n, in_channels=player.model.in_channels),
                                 field=validation_field, eps=1., device=device)

    # model (x) vs random (o):
    scores = np.array([play_duel(player_x=player_model_0, player_o=player_random, return_result_only=True)
                       for _ in range(n_games)])
    logger.add_scalar('val/winrate/model(x) vs random(o)', np.mean(scores == 1), epoch)

    # random (x) vs model (o):
    scores = np.array([play_duel(player_x=player_random, player_o=player_model_0, return_result_only=True)
                       for _ in range(n_games)])
    logger.add_scalar('val/winrate/model(o) vs random(x)', np.mean(scores == -1), epoch)

    # model (x) vs model (o):
    scores = np.array([play_duel(player_x=player_model_0, player_o=player_model_1, return_result_only=True)
                       for _ in range(n_games)])
    logger.add_scalar('val/winrate/model(x) vs model(o)', np.mean(scores == 1), epoch)

    # model (x) vs opponent (o):
    # opponent (x) vs model (o):
    if opponent_model_path is not None:
        player_opponent = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps,
                                       device=device)
        load_model_state(module=player_opponent.model, path=opponent_model_path)

        scores = np.concatenate((
            np.array([play_duel(player_x=player_model_0, player_o=player_opponent, return_result_only=True)
                      for _ in range(n_games)]),
            -np.array([play_duel(player_x=player_opponent, player_o=player_model_0, return_result_only=True)
                       for _ in range(n_games)])
        ))

        logger.add_scalar('val/winrate/model vs opponent', np.mean(scores == 1), epoch)

        del player_opponent

        if return_winrate_vs_opponent:
            del player_random, player_model_0, player_model_1, validation_field
            return np.mean(scores == 1)

    del player_random, player_model_0, player_model_1, validation_field
