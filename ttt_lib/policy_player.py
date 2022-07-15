import numpy as np
import torch

from dpipe.torch import to_device, to_var, to_np


class PolicyPlayer:
    def __init__(self, model, field, eps=0.0, device='cpu'):
        self.model = to_device(model, device=device)
        self.field = field
        self.eps = eps
        self.device = device

        # fields to slightly speedup computation:
        self.n = self.field.get_n()

        self.eval()

    def _forward_action(self, i, j):
        self.field.make_move(i, j)
        return self.field.get_value()

    def _retract_action(self, i, j):
        self.field.retract_move(i, j)
        return None

    def _calc_move(self, policy, eps):
        """
        eps = None  : proba choice
        eps = 0     : greedy choice
        eps > 0     : eps-greedy choice
        """
        avail_a = torch.nonzero(self.field.get_running_features()[:, -1, ...].reshape(-1)).reshape(-1)
        n_a = len(avail_a)

        avail_p = policy.reshape(-1)[avail_a]

        # greedy choice:
        a_idx = torch.argmax(avail_p).item()
        exploit = True

        if eps is None:  # proba choice:
            _a_idx = np.random.choice(np.arange(n_a), p=to_np(avail_p))
            exploit = (_a_idx == a_idx)
            a_idx = _a_idx
        else:
            if eps > 0:  # eps-greedy
                soft_p = np.zeros(n_a) + eps / n_a
                soft_p[a_idx] += (1 - eps)
                _a_idx = np.random.choice(np.arange(n_a), p=soft_p)

                exploit = (_a_idx == a_idx)
                a_idx = _a_idx

        a = avail_a[a_idx].item()
        return a, exploit

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

    def forward_state(self):
        state = self.field.get_running_features()
        policy = self.forward(state)
        proba = self.model.predict_proba(state, policy)
        return policy[0][0], proba[0][0]

    def action(self, train=True):
        eps = self.eps
        policy, proba = self.forward_state()

        if eps is None:
            action, exploit = self._calc_move(policy=proba, eps=eps)
        else:
            action, exploit = self._calc_move(policy=policy, eps=eps)

        value = self._forward_action(*self.a2ij(action))
        return policy, proba, action, exploit, value

    def ucb_action(self, q_est, u, q_est_skip_value=-2, lam=0.5):
        """ Action interface for MCTS. """
        policy, proba = self.forward_state()

        mask_est = q_est != q_est_skip_value
        policy[mask_est] = lam * policy[mask_est] + (1 - lam) * q_est[mask_est]
        policy = policy + proba * u

        action, exploit = self._calc_move(policy=policy, eps=0)
        value = self._forward_action(*self.a2ij(action))
        return policy, proba, action, exploit, value

    def manual_action(self, i, j):
        # compatibility with `action` output:
        action = self.ij2a(i, j)
        policy_manual = to_var(np.zeros((self.n, self.n), dtype='float32'), device=self.device)
        policy_manual[i, j] = 1
        proba_manual = to_var(np.zeros((self.n, self.n), dtype='float32'), device=self.device)
        proba_manual[i, j] = 1

        value = self._forward_action(i, j)

        return policy_manual, proba_manual, action, None, value

    def update_field(self, field):
        self.field.set_field(field)
