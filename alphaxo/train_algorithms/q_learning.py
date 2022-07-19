# TODO: should be revised!


import numpy as np
from tqdm import trange
import torch
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
from dpipe.io import PathLike
from dpipe.torch import to_var, save_model_state

from alphaxo.self_games import play_self_game
from alphaxo.validate import validate
from alphaxo.policy_player import PolicyPlayer
from alphaxo.torch.model import optimizer_step


def train_q_learning(player: PolicyPlayer, logger: SummaryWriter, exp_path: PathLike, n_episodes: int,
                     augm: bool = True, n_step_q: int = 2, lam: float = None, sigma_p: float = 0.5,
                     ep2eps: dict = None, lr: float = 4e-3,
                     episodes_per_epoch: int = 10000, n_duels: int = 1000, episodes_per_model_save: int = 100000,
                     duel_path: PathLike = None):
    """
    Params
    ------
    player: PolicyPlayer
    logger: SummaryWriter
    exp_path: PathLike
    n_episodes: int
    augm: bool, default `True`.
    n_step_q: int. The depth of action-value back-propagation through MC tree. The same as n in n-step tree-backup.
        If `n_step_q=0`, `n_step_q` is being drawn from the Poisson distribution with `lam=lam` for the episode.
        If `n_step_q=-1`, `n_step_q` is being drawn from the Poisson distribution with `lam=lam` for every iteration.
    lam: float
    sigma: float, optional. # TODO: is not implemented yet.
    ep2eps: dict
    lr: float
    episodes_per_epoch: int
    n_duels: int
    episodes_per_model_save: int
    duel_path: PathLike

    Returns
    -------
    None
    """

    n = player.field.get_size()
    optimizer = SGD(player.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)

    ep2eps_curr_idx = None if ep2eps is None else 0
    ep2eps_items = None if ep2eps is None else list(ep2eps.items())
    for ep in trange(n_episodes):

        s_history, f_history, a_history, q_history, q_max_history, p_history, e_history, value\
            = play_self_game(player=player, augm=augm)

        sigmas = np.random.binomial(size=len(a_history), n=1, p=sigma_p)

        if n_step_q > 0:
            _n_step_qs = [n_step_q, ] * len(a_history)
        elif n_step_q == 0:
            _n_step_qs = [1 + np.random.poisson(lam=lam), ] * len(a_history)
        else:  # n_step_q == -1:  # TODO: also works for n_step_q < -1
            _n_step_qs = np.random.poisson(lam=3, size=len(a_history)) + 1

        player.train()
        qs_pred = player.forward(to_var(np.concatenate(f_history, axis=0), device=player.device)).squeeze(1)

        loss = torch.tensor(0., requires_grad=True, dtype=torch.float32)
        loss.to(player.device)

        rev_a_history = a_history[::-1]
        rev_e_history = e_history[::-1]
        rev_q_history = torch.flip(qs_pred, dims=(0, ))
        rev_q_max_history = q_max_history[::-1]

        for t_rev, (a, q, _n_step_q) in enumerate(zip(rev_a_history, rev_q_history, _n_step_qs)):
            q_star = 0
            for k in range(max(0, t_rev - _n_step_q + 1), t_rev + 1):
                if k == 0:
                    q_star = value
                else:
                    n_actions = n ** 2 - len(a_history) + t_rev + 1

                    pi = 1 if rev_e_history[k] else 0
                    rho = n_actions / (n_actions * (1 - player.eps) + player.eps) if rev_e_history[k] else 0

                    q_star = - (rev_q_max_history[k]
                                + (sigmas[k] * rho + (1 - sigmas[k]) * pi) * (q_star - rev_q_max_history[k]))

            loss = loss + .5 * (q[a // n, a % n] - q_star) ** 2

        optimizer_step(optimizer=optimizer, loss=loss)

        # ### logging: ###
        logger.add_scalar('loss/train', loss.item(), ep)

        # ### saving model: ###
        if (ep > 0) and (ep % episodes_per_model_save == 0):
            save_model_state(player.model, exp_path / f'model_{ep}.pth')

        # ### validation: ###
        if (ep > 0) and (ep % episodes_per_epoch == 0):
            validate(val=ep // episodes_per_epoch, ep=ep, player=player, logger=logger, n=n, n_duels=n_duels,
                     duel_path=duel_path)

        # ### scheduler(eps): ###
        if ep2eps_curr_idx is not None:
            if ep2eps_curr_idx < len(ep2eps_items):
                if ep == ep2eps_items[ep2eps_curr_idx][0]:
                    player.eps = ep2eps_items[ep2eps_curr_idx][1]
                    ep2eps_curr_idx += 1
