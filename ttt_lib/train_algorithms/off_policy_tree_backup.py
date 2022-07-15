import os
from pathlib import Path

import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike
from dpipe.torch import to_var, save_model_state, load_model_state

from ttt_lib.field import X_ID
from ttt_lib.policy_player import PolicyPlayer
from ttt_lib.self_games import play_duel
from ttt_lib.torch.model import optimizer_step
from ttt_lib.utils import get_random_field
from ttt_lib.validate import validate


def train_tree_backup(player: PolicyPlayer, opponent: PolicyPlayer, logger: SummaryWriter, models_bank_path: PathLike,
                      n_episodes: int, n_step_q: int = 4, episodes_per_epoch: int = 10000, n_val_games: int = 400,
                      random_starts: bool = False, random_starts_max_depth: int = 10,
                      lr_init: float = 4e-3, epoch2lr: dict = None, epoch2eps: dict = None,
                      best_model_name: str = 'model.pth', winrate_th: float = 0.55):
    # ### train init: ###
    models_bank_path = Path(models_bank_path)
    best_model_path = models_bank_path / best_model_name
    if not best_model_path.exists():
        save_model_state(player.model, best_model_path)

    n = player.field.get_n()

    optimizer = SGD(player.model.parameters(), lr=lr_init, momentum=0.9, weight_decay=1e-5, nesterov=True)

    # ### train loop: ###
    prev_epoch = -1
    for ep in trange(n_episodes):

        # ### 1. random starts ###
        init_field = None
        if random_starts:
            if np.random.random_sample() < player.eps:
                init_field = get_random_field(n=n, min_depth=0, max_depth=random_starts_max_depth)

        # ### 2. sampling a game ###
        is_player_x = bool(np.random.randint(2))
        player_x, player_o = (player, opponent) if is_player_x else (opponent, player)
        s_history, f_history, a_history, q_history, q_max_history, p_history, e_history, winner\
            = play_duel(player_x=player_x, player_o=player_o, field=init_field)
        value = winner ** 2

        if len(s_history) == 0:
            continue

        if is_player_x != (player_x.field.get_opponent_action_id() == X_ID):
            s_history, f_history, a_history, q_history, q_max_history, p_history, e_history\
                = strip_arrs(s_history, f_history, a_history, q_history, q_max_history, p_history, e_history)
            value = -value

            if len(s_history) == 0:
                continue

        rev_f_history, rev_a_history, rev_q_max_history\
            = rev_history_for_the_last_player(f_history, a_history, q_max_history)

        # ### 3. forward step for all encountered states ###
        player.train()
        rev_q_history = player.forward(to_var(np.concatenate(rev_f_history, axis=0), device=player.device)).squeeze(1)

        # ### 4. calculating loss with TB algorithm ###
        loss = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        loss.to(player.device)

        for t_rev, (a, q) in enumerate(zip(rev_a_history, rev_q_history)):
            if t_rev < n_step_q:
                q_star = (-1) ** t_rev * value
            else:
                q_star = (-1) ** n_step_q * rev_q_max_history[t_rev - n_step_q]
            loss = loss + .5 * (q[a // n, a % n] - q_star) ** 2

        # ### 5. gradient step ###
        optimizer_step(optimizer=optimizer, loss=loss)

        # ###### validation and epoch switching ######
        logger.add_scalar('loss/train', loss.item(), ep)

        epoch = ep // episodes_per_epoch
        if epoch > prev_epoch:
            prev_epoch = epoch

            # ### save current model to the bank ###
            save_model_state(player.model, models_bank_path / f'model_{epoch}.pth')

            # ### 6. load a random opponent from the bank (for the next epoch) ###
            load_random_opponent(opponent=opponent, models_bank_path=models_bank_path)

            # ### validation ###
            player_eps = player.eps
            player.eps = None  # proba choice
            winrate_vs_best = validate(epoch=epoch, player=player, logger=logger, n=n, n_games=n_val_games,
                                       opponent_model_path=best_model_path, return_winrate_vs_opponent=True)
            player.eps = player_eps

            if winrate_vs_best >= winrate_th:
                save_model_state(player.model, best_model_path)

            # ### schedulers ###
            try:
                player.eps = epoch2eps[epoch]
            except (KeyError, TypeError):  # (no such epochs in the dict, dict is None)
                pass

            try:
                optimizer.param_groups[0]['lr'] = epoch2lr[epoch]
            except (KeyError, TypeError):  # (no such epochs in the dict, dict is None)
                pass


def load_random_opponent(opponent: PolicyPlayer, models_bank_path: PathLike):
    model_paths = [p for p in Path(models_bank_path).glob('*.pth')]
    if len(model_paths) > 0:
        load_model_state(opponent.model, np.random.choice(model_paths))
    opponent.eval()


def rev_history_for_the_last_player(*histories):
    return [history[::-2] for history in histories]


def strip_arrs(*arrs):
    return [arr[:-1] for arr in arrs]
