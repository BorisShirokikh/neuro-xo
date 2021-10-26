import random
from copy import deepcopy

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from dpipe.torch import to_device, to_var

from ttt_lib.field import Field
from ttt_lib.im.augm import augm_spatial
from ttt_lib.torch.module.policy_net import PolicyNetworkRandom
from ttt_lib.torch.utils import wait_and_load_model_state


class PolicyPlayer:
    def __init__(self, model, field, eps=0.0, device='cpu'):
        self.model = to_device(model, device=device)
        self.field = field
        self.eps = eps
        self.device = device

        # fields to slightly speedup computation:
        self.n = self.field.get_size()

        self.eval()

    def _forward_action(self, i, j):
        self.field.make_move(i, j)
        return self.field.get_value()

    def _backward_action(self, i, j):
        self.field.backward_move(i, j)
        return None

    def _calc_move(self, policy, eps):
        avail_actions = torch.nonzero(self.field.get_running_features()[:, -1, ...].reshape(-1)).reshape(-1)
        n_actions = len(avail_actions)

        avail_p = policy.reshape(-1)[avail_actions]
        argmax_avail_action_idx = random.choice(torch.where(avail_p == avail_p.max(), 1., 0.).nonzero()).item()
        exploit = True

        if eps > 0:
            soft_p = np.zeros(n_actions) + eps / n_actions
            soft_p[argmax_avail_action_idx] += (1 - eps)
            soft_p_action_idx = np.random.choice(np.arange(n_actions), p=soft_p)
            exploit = (soft_p_action_idx == argmax_avail_action_idx)
            argmax_avail_action_idx = soft_p_action_idx

        action = avail_actions[argmax_avail_action_idx].item()

        return action, action // self.n, action % self.n, exploit

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def forward_state(self, state=None):
        state = self.field.get_running_features() if state is None else self.field.field2features(field=state)
        policy = self.forward(state)
        proba = self.model.predict_proba(state, policy)
        return policy[0][0], proba[0][0]

    def action(self, train=True):
        eps = self.eps if train else 0

        policy, proba = self.forward_state()

        action, i, j, exploit = self._calc_move(policy=policy, eps=eps)
        value = self._forward_action(i, j)

        return policy, proba, action, exploit, value

    def manual_action(self, i, j):
        # compatibility with `action` output:
        action = i * self.n + j
        policy_manual = to_var(np.zeros((self.n, self.n), dtype='float32'), device=self.device)
        policy_manual[i, j] = 1
        proba_manual = to_var(np.zeros((self.n, self.n), dtype='float32'), device=self.device)
        proba_manual[i, j] = 1

        value = self._forward_action(i, j)

        return policy_manual, proba_manual, action, None, value

    def update_field(self, field):
        self.field.set_state(field=field)


def play_self_game(player, field=None, train=True, augm=True):
    if field is None:
        field = np.zeros((player.field.get_size(), ) * 2, dtype='float32')
    player.update_field(field=field)

    if not train:
        augm = False

    s_history = []      # states
    f_history = []      # features
    a_history = []      # actions
    e_history = []      # exploits
    q_history = []      # policies
    q_max_history = []  # max Q values
    p_history = []      # probas

    player.eval()
    v = None  # v -- value
    while v is None:
        s_history.append(player.field.get_state())
        f_history.append(player.field.get_features())
        q, p, a, e, v = player.action(train=train)
        q_history.append(q)
        q_max_history.append(q[p > 0].max().item())
        p_history.append(p)
        a_history.append(a)
        e_history.append(e)

        if augm:
            player.update_field(augm_spatial(player.field.get_state()))

    return s_history, f_history, a_history, q_history, q_max_history, p_history, e_history, v


def play_duel(player_x, player_o, field=None, return_result_only=False):
    if field is None:
        field = np.zeros((player_x.field.get_size(), ) * 2, dtype='float32')
    player_x.update_field(field=field)
    player_o.update_field(field=field)

    is_x_next = player_x.field.next_action_id == 1
    player_act = player_x if is_x_next else player_o
    player_wait = player_o if is_x_next else player_x

    s_history = []  # states
    f_history = []  # features
    a_history = []  # actions
    e_history = []  # exploitations
    q_history = []  # policies
    p_history = []  # probas

    player_act.eval()
    player_wait.eval()
    v = None  # v -- value
    while v is None:
        s_history.append(player_act.field.get_state())
        f_history.append(player_act.field.get_features())
        q, p, a, e, v = player_act.action(train=True)
        q_history.append(q)
        p_history.append(p)
        a_history.append(a)
        e_history.append(e)

        if v is None:
            player_wait.update_field(field=player_act.field.get_state())
            player_temp = player_act
            player_act = player_wait
            player_wait = player_temp

    if v == 0:
        winner = 0
    else:  # v == 1:
        winner = - player_act.field.next_action_id

    return winner if return_result_only else (s_history, f_history, a_history, q_history, p_history, e_history, winner)


def validate(val: int, ep: int, player: PolicyPlayer, logger: SummaryWriter, n: int, n_duels: int = 1000,
             duel_path=None):
    device = player.device

    validation_field = Field(n=n, kernel_len=player.field.kernel_len, device=device, check_device=device)

    player_model = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)
    player_opponent = PolicyPlayer(model=deepcopy(player.model), field=validation_field, eps=player.eps, device=device)
    if duel_path is not None:
        wait_and_load_model_state(module=player_opponent.model, exp_path=duel_path, ep=ep)

    player_random = PolicyPlayer(model=PolicyNetworkRandom(n=n, in_channels=player.model.in_channels),
                                 field=validation_field, eps=1., device=device)

    # model (x) vs random (o):
    scores = np.array([play_duel(player_x=player_model, player_o=player_random, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(x) vs random(o)', np.mean(scores == 1), val)
    logger.add_scalar('val/draw_fraction/model(x) vs random(o)', np.mean(scores == 0), val)

    # random (x) vs model (o):
    scores = np.array([play_duel(player_x=player_random, player_o=player_model, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(o) vs random(x)', np.mean(scores == -1), val)
    logger.add_scalar('val/draw_fraction/model(o) vs random(x)', np.mean(scores == 0), val)

    # model (x) vs model (o):
    scores = np.array([play_duel(player_x=player, player_o=player_model, return_result_only=True)
                       for _ in range(n_duels)])
    logger.add_scalar('val/winrate/model(x) vs model(o)', np.mean(scores == 1), val)
    logger.add_scalar('val/draw_fraction/model(x) vs model(o)', np.mean(scores == 0), val)
    logger.add_scalar('val/winrate/model(o) vs model(x)', np.mean(scores == -1), val)

    # model (x) vs opponent (o):
    # opponent (x) vs model (o):
    if duel_path is not None:
        scores = np.array([play_duel(player_x=player_model, player_o=player_opponent, return_result_only=True)
                           for _ in range(n_duels)])
        logger.add_scalar('val/winrate/model(x) vs opponent(o)', np.mean(scores == 1), val)
        logger.add_scalar('val/draw_fraction/model(x) vs opponent(o)', np.mean(scores == 0), val)
        logger.add_scalar('val/winrate/opponent(o) vs model(x)', np.mean(scores == -1), val)

        scores = np.array([play_duel(player_x=player_model, player_o=player_opponent, return_result_only=True)
                           for _ in range(n_duels)])
        logger.add_scalar('val/winrate/opponent(x) vs model(o)', np.mean(scores == 1), val)
        logger.add_scalar('val/draw_fraction/opponent(x) vs model(o)', np.mean(scores == 0), val)
        logger.add_scalar('val/winrate/model(o) vs opponent(x)', np.mean(scores == -1), val)

    del player_random, player_model, player_opponent, validation_field