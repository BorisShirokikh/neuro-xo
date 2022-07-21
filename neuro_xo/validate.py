from copy import deepcopy

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from neuro_xo.field import Field
from neuro_xo.policy_player import PolicyPlayer
from neuro_xo.self_games import play_duel
from neuro_xo.torch.module.policy_net import PolicyNetworkRandom
from neuro_xo.torch.utils import wait_and_load_model_state


def validate(val: int, ep: int, player: PolicyPlayer, logger: SummaryWriter, n: int, n_duels: int = 1000,
             duel_path=None):
    device = player.device

    validation_field = Field(n=n, kernel_len=player.field.kernel_len, device=device, check_device=device)

    player_model_0 = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)
    player_model_1 = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)

    player_random = PolicyPlayer(model=PolicyNetworkRandom(n=n, in_channels=player.model.in_channels),
                                 field=validation_field, eps=1., device=device)

    # model (x) vs random (o):
    scores = np.array([play_duel(player_x=player_model_0, player_o=player_random, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(x) vs random(o)', np.mean(scores == 1), val)
    logger.add_scalar('val/draw_fraction/model(x) vs random(o)', np.mean(scores == 0), val)

    # random (x) vs model (o):
    scores = np.array([play_duel(player_x=player_random, player_o=player_model_0, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(o) vs random(x)', np.mean(scores == -1), val)
    logger.add_scalar('val/draw_fraction/model(o) vs random(x)', np.mean(scores == 0), val)

    # model (x) vs model (o):
    scores = np.array([play_duel(player_x=player_model_0, player_o=player_model_1, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(x) vs model(o)', np.mean(scores == 1), val)
    logger.add_scalar('val/draw_fraction/model(x) vs model(o)', np.mean(scores == 0), val)
    logger.add_scalar('val/winrate/model(o) vs model(x)', np.mean(scores == -1), val)

    # model (x) vs opponent (o):
    # opponent (x) vs model (o):
    if duel_path is not None:
        player_opponent = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps,
                                       device=device)
        wait_and_load_model_state(module=player_opponent.model, exp_path=duel_path, ep=ep)

        scores = np.array([play_duel(player_x=player_model_0, player_o=player_opponent, return_result_only=True)
                           for _ in range(n_duels)])
        logger.add_scalar('val/winrate/model(x) vs opponent(o)', np.mean(scores == 1), val)
        logger.add_scalar('val/draw_fraction/model(x) vs opponent(o)', np.mean(scores == 0), val)
        logger.add_scalar('val/winrate/opponent(o) vs model(x)', np.mean(scores == -1), val)

        scores = np.array([play_duel(player_x=player_model_0, player_o=player_opponent, return_result_only=True)
                           for _ in range(n_duels)])
        logger.add_scalar('val/winrate/opponent(x) vs model(o)', np.mean(scores == 1), val)
        logger.add_scalar('val/draw_fraction/opponent(x) vs model(o)', np.mean(scores == 0), val)
        logger.add_scalar('val/winrate/model(o) vs opponent(x)', np.mean(scores == -1), val)

        del player_opponent

    del player_random, player_model_0, player_model_1, validation_field
