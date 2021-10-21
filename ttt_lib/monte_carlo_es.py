import random

import numpy as np
from tqdm.notebook import trange
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import binary_cross_entropy
from dpipe.torch import to_device, to_var
from dpipe.layers import ResBlock2d

from ttt_lib.utils import get_running_mean_hash_table, update_running_mean_hash_table, \
    sort_and_clear_running_mean_hash_table, make_state_hashable, augm_spatial
from ttt_lib.torch import optimizer_step, Bias, MaskedSoftmax


class PolicyNetworkMCES(nn.Module):
    def __init__(self, structure=None, n=10, m=10, in_channels=5):
        super().__init__()
        if structure is None:
            structure = (16, 32)
        self.structure = structure

        self.n = n
        self.m = m

        self.in_channels = in_channels

        n_features = structure[0]
        n_features_policy = structure[1]
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, n_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=False),
            nn.Flatten(),
            Bias(n_channels=n * m),
            # nn.Softmax(dim=1),
        )
        self.flatten = nn.Flatten()
        self.masked_softmax = MaskedSoftmax(dim=1)

    def forward(self, x):
        return torch.reshape(self.masked_softmax(self.model(x), self.flatten(x[:, -1, ...].unsqueeze(1))),
                             shape=(-1, 1, self.n, self.m))


class PolicyPlayer:
    def __init__(self, model, field, eps=0.1, device='cpu'):
        self.model = to_device(model, device=device)
        self.field = field
        self.eps = eps
        self.device = device

        # fields to slightly speedup computation:
        self.n, self.m = self.field.get_shape()

        self.eval()

    def _forward_action(self, i, j):
        self.field.make_move(i, j)
        return self.field.get_value()

    def _calc_move(self, policy, eps):
        avail_actions = torch.nonzero(self.field.get_running_features()[:, -1, ...].ravel()).ravel()
        n_actions = len(avail_actions)

        avail_p = policy.ravel()[avail_actions]
        argmax_avail_action_idx = random.choice(torch.where(avail_p == avail_p.max(), 1., 0.).nonzero()).item()

        if eps > 0:
            soft_p = np.zeros(n_actions) + eps / n_actions
            soft_p[argmax_avail_action_idx] += (1 - eps)
            argmax_avail_action_idx = np.random.choice(np.arange(n_actions), p=soft_p)

        action = avail_actions[argmax_avail_action_idx].item()

        return action, action // self.m, action % self.m

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def forward_state(self, state=None):
        state = self.field.get_running_features() if state is None else self.field.field2features(field=state)
        return self.forward(state)[0][0]

    def action(self, train=True):
        eps = self.eps if train else 0

        policy = self.forward_state()

        action, i, j = self._calc_move(policy=policy, eps=eps)
        value = self._forward_action(i, j)

        return policy, action, value

    def manual_action(self, i, j):
        # compatibility with `action` output:
        action = i * self.m + j
        policy_manual = to_var(np.zeros((self.n, self.m), dtype='float32'), device=self.device)
        policy_manual[i, j] = 1

        value = self._forward_action(i, j)

        return policy_manual, action, value

    def update_field(self, field):
        self.field.set_state(field=field)


def play_game(player, field=None, train=True, augm=False, return_p=False):
    if field is None:
        field = np.zeros(player.field.get_shape(), dtype='float32')
    player.update_field(field=field)

    if not train:
        augm = False

    s_history = []
    f_history = []
    a_history = []
    p_history = []

    player.eval()
    v = None  # v -- value
    while v is None:
        s_history.append(player.field.get_state())
        f_history.append(player.field.get_features())
        p, a, v = player.action(train=train)
        a_history.append(a)
        p_history.append(p)

        if augm:
            player.update_field(augm_spatial(player.field.get_state()))

    return (f_history, s_history, a_history, v, p_history) if return_p else (f_history, s_history, a_history, v)


def backprop_sa(p_pred, state, action, value, sa2g, m, key_m='running_mean', key_n='n'):
    p_true = torch.zeros_like(p_pred)

    hashable_state = make_state_hashable(state)

    available_actions = np.nonzero(state.ravel() == 0)[0]
    for a in available_actions:
        h = hash((hashable_state, a))
        i, j = a // m, a % m

        if a == action:
            update_running_mean_hash_table(ht=sa2g, h=h, v=value, key_m=key_m, key_n=key_n)

        if sa2g[h][key_n] > 0:
            p_true[i, j] = sa2g[h][key_m]
        else:
            p_true[i, j] = p_pred[i, j]

    argmax_action = random.choice(torch.where(p_true == p_true.max(), 1., 0.).nonzero())
    p_true = torch.zeros_like(p_pred)
    p_true[argmax_action[0], argmax_action[1]] = 1

    return p_true


def train_on_policy_monte_carlo_es(player, n_episodes=1000, eps=0.1, lr=1e-2, hash_expiration_pool=100000,
                                   verbose_module=trange, augm=False, logger=None):
    optimizer = Adam(player.model.parameters(), lr=lr)
    criterion = binary_cross_entropy

    sa2g = get_running_mean_hash_table()

    n, m = player.field.get_shape()

    for ep in verbose_module(n_episodes):
        player.eval()
        f_history, s_history, a_history, value = play_game(player=player, augm=augm)

        player.train()

        ps_pred = player.forward(to_var(np.concatenate(f_history, axis=0), device=player.device)).squeeze(1)
        ps_true = []

        v = value
        for s, a, p in zip(s_history[::-1], a_history[::-1], torch.flip(ps_pred, dims=(0, ))):
            p_true = backprop_sa(p_pred=p, state=s, action=a, value=v, sa2g=sa2g, m=m)
            v = 1 - v
            ps_true.append(p_true[None])

        ps_true = to_device(torch.cat(ps_true[::-1], dim=0), device=player.device)

        loss = criterion(ps_pred, ps_true)
        optimizer_step(optimizer=optimizer, loss=loss)
        logger.add_scalar('loss/train', loss.item(), ep)

        player.update_field(field=np.zeros((n, m)))

        if len(sa2g) > hash_expiration_pool:
            sa2g = sort_and_clear_running_mean_hash_table(sa2g, hashes_to_leave=int(hash_expiration_pool * (1 - eps)))

    return sa2g
