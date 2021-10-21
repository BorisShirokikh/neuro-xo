import numpy as np
import torch
import torch.nn as nn
from rl_ttt.field import Field
from rl_ttt.model.torch import optimizer_step
from torch.nn.functional import binary_cross_entropy
from tqdm.notebook import trange
from dpipe.torch import to_np, to_device
from dpipe.layers import ResBlock2d
from torch.optim import Adam

from rl_ttt.model.utils import get_running_mean_hash_table, field2state, update_running_mean_hash_table, \
    sort_and_clear_running_mean_hash_table


class PolicyNetworkMCES(nn.Module):
    def __init__(self, structure=None, n=10, m=10):
        super().__init__()
        self.n = n
        self.m = m

        if structure is None:
            structure = (16, 32)

        n_features = structure[0]
        n_features_policy = structure[1]
        self.model = nn.Sequential(
            nn.Conv2d(1, n_features, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            ResBlock2d(n_features, n_features, kernel_size=3, padding=1, batch_norm_module=nn.Identity),
            # policy head
            nn.Conv2d(n_features, n_features_policy, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_features_policy, 1, kernel_size=1, padding=0, bias=True),
            nn.Flatten(),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return torch.reshape(self.model(x), shape=(-1, 1, self.n, self.m))


class PolicyPlayer:
    def __init__(self, model, field, train_eps=0.1, eval_eps=0.01, device='cpu'):
        self.model = to_device(model, device=device)
        self.field = field
        self.train_eps = train_eps
        self.eval_eps = eval_eps
        self.device = device

        self.action_id = 1

        # fields to slightly speedup computation:
        self.n, self.m = self.field.get_shape()

    def update_field(self, field):
        self.field.set_state(field=field)
        self.action_id = 1 if (self.field.get_depth() % 2 == 0) else -1

    def _update_action_id(self):
        self.action_id = -self.action_id

    def _forward_action(self, i, j):
        value = None
        try:
            self.field.make_move(i, j, self.action_id)
        except IndexError:  # from `make_move`: "Accessing already marked cell."
            value = 0
        if self.field.check_win(self.action_id):
            value = 1
        elif self.field.check_draw():
            value = 0.5

        self._update_action_id()

        return value

    def _calc_move(self, p, s, eps):
        avail_actions = torch.nonzero(s.ravel() == 0).ravel()
        n_actions = avail_actions.nelement()

        avail_p = p.ravel()[avail_actions]
        action_idx = avail_p.argmax().item()

        if eps > 0:
            soft_p = np.zeros(n_actions) + eps / n_actions
            soft_p[action_idx] += (1 - eps)
            action_idx = np.random.choice(np.arange(n_actions), p=soft_p)

        action = avail_actions[action_idx]

        return action, action // self.m, action % self.m

    def forward_state(self, state):
        self.model.train()
        state = to_device(state[None, None], device=self.field.device)
        return self.model(state)[0][0]

    def action(self, train=True):
        eps = self.train_eps if train else self.eval_eps
        self.model.eval()

        state = to_device(torch.Tensor(self.field.get_state()), device=self.field.device)
        policy = self.model(state[None, None])[0][0]

        action, i, j = self._calc_move(p=policy, s=state, eps=eps)
        value = self._forward_action(i, j)

        return state, policy, action, value

    def manual_action(self, i, j):
        self.model.eval()

        # compatibility with `act` output:
        action = i * self.m + j
        policy_manual = to_device(torch.zeros(self.n, self.m), device=self.field.device)

        value = self._forward_action(i, j)
        state = to_device(torch.Tensor(self.field.get_state()), device=self.field.device)

        return state, policy_manual, action, value

    # def forward_states(self, states):
    #     self.model.train()
    #     states = to_device(torch.cat([s[None, None] for s in states]), device=self.field.device)
    #     return self.model(states)


def play_game(player, field=None):
    if field is None:
        field = np.zeros(player.field.get_shape())
    player.update_field(field=field)

    s_history = []
    a_history = []

    v = None  # v -- value
    while v is None:
        s, p, a, v = player.action()
        s_history.append(s)
        a_history.append(a)

    return s_history, a_history, v


def backprop_action_value(field_backward, action, sa2g, value, backward=True, key_m='running_mean'):
    state = field2state(field_backward)
    h = hash((state, action))
    update_running_mean_hash_table(ht=sa2g, h=h, v=value, key_m=key_m)

    if backward:
        _, m, = field_backward.get_shape()
        i, j = action // m, action % m
        field_backward.backward_move(i, j)

    return sa2g[h][key_m]


def develop_state(player, state, real_action, real_g, sa2g):
    n, m = player.n, player.m
    p_true = torch.zeros(n, m)

    i, j = real_action // m, real_action % m
    p_true[i, j] = real_g

    avail_actions = torch.nonzero(state.ravel() == 0).ravel()

    for developed_action in avail_actions:
        if developed_action != real_action:

            player.update_field(state)
            i, j = developed_action // m, developed_action % m
            player.manual_action(i, j)

            s_history, a_history, value = play_game(player=player, field=player.field.get_state())

            field_backward = Field(n=n, m=m, field=player.field.get_state(), device=player.field.device)

            v = value
            for s, a in zip(s_history[::-1], a_history[::-1]):
                g = backprop_action_value(field_backward=field_backward, action=a, sa2g=sa2g, value=v)
                v = 1 - v

            p_true[i, j] = g

    argmax_action = p_true.argmax().item()
    i, j = argmax_action // m, argmax_action % m
    p_true = torch.zeros(n, m)
    p_true[i, j] = 1

    return to_device(p_true, device=player.field.device)


def train_on_policy_monte_carlo_es(player, n_episodes=1000, train_eps=0.1, lr=1e-2,
                                   hash_expiration_pool=100000):
    loss_history = []
    optimizer = Adam(player.model.parameters(), lr=lr)

    sa2g = get_running_mean_hash_table()

    for _ in trange(n_episodes):
        s_history, a_history, value = play_game(player=player)

        field = player.field
        n, m = field.get_shape()

        field_backward = Field(n=n, m=m, field=field.get_state(), device=field.device)

        v = value
        for s, a in zip(s_history[::-1], a_history[::-1]):
            g = backprop_action_value(field_backward=field_backward, action=a, sa2g=sa2g, value=v)
            p_true = develop_state(player, state=s, real_action=a, real_g=g, sa2g=sa2g)
            p_pred = player.forward_state(s)
            loss = binary_cross_entropy(p_pred, p_true)
            optimizer_step(optimizer=optimizer, loss=loss)

            loss_history.append(loss.item())
            v = 1 - v

        player.update_field(field=np.zeros((n, m)))

        if len(sa2g) > hash_expiration_pool:
            sa2g = sort_and_clear_running_mean_hash_table(sa2g, hashes_to_leave=int(hash_expiration_pool * train_eps))

    return loss_history, sa2g
