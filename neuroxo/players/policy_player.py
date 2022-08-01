import numpy as np
import torch

from dpipe.torch import to_device, to_np

from neuroxo.torch.utils import get_available_moves


class PolicyPlayer:
    def __init__(self, model, field, eps=0.0, device='cpu'):
        self.model = to_device(model, device=device)
        self.field = field
        self.eps = eps
        self.device = device

        # fields to slightly speedup computation:
        self.n = self.field.get_n()

        self.eval()

    def _make_move(self, i, j):
        self.field.make_move(i, j)

    def _calc_move(self, policy, eps):
        """
        eps = None  : proba choice
        eps = 0     : greedy choice
        eps > 0     : eps-greedy choice
        """
        avail_a = torch.nonzero(get_available_moves(self.field.get_running_features()).reshape(-1)).reshape(-1)
        n_a = len(avail_a)

        avail_p = policy.reshape(-1)[avail_a]

        # greedy choice:
        a_idx = torch.argmax(avail_p).item()

        if eps is None:  # proba choice:
            _a_idx = np.random.choice(np.arange(n_a), p=to_np(avail_p))
            a_idx = _a_idx
        else:
            if eps > 0:  # eps-greedy
                soft_p = np.zeros(n_a) + eps / n_a
                soft_p[a_idx] += (1 - eps)
                _a_idx = np.random.choice(np.arange(n_a), p=soft_p)

                a_idx = _a_idx

        a = avail_a[a_idx].item()
        return a

    def a2ij(self, a):
        return a // self.n, a % self.n

    def ij2a(self, i, j):
        return i * self.n + j

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def forward(self, x):
        return self.model(x)

    def predict_action_values(self, logits):
        return self.model.predict_action_values(logits)

    def forward_state(self):

        state = self.field.get_running_features()
        logits = self.forward(state)
        policy = self.predict_action_values(logits)
        proba = self.model.predict_proba(state, logits)
        return policy[0][0], proba[0][0]

    def action(self):
        eps = self.eps
        proba, policy = self.model(self.field.get_running_features())
        proba, policy = proba[0][0], policy[0][0]
        action = self._calc_move(policy=proba, eps=eps) if (eps is None) else self._calc_move(policy=policy, eps=eps)

        self._make_move(*self.a2ij(action))

        return action, proba, policy

    def on_opponent_action(self, a: int):
        pass

    def update_state(self, field):
        self.field.set_field(field)

    def set_field(self, field):
        self.field = field
