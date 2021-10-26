import random

import numpy as np
import torch
from dpipe.torch import to_device, to_var


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

    def forward_state(self, state=None, tta=False):
        state = self.field.get_running_features() if state is None else self.field.field2features(field=state)

        if tta:
            augm_state = []
            for k in (0, 1, 2, 3):
                sr = state.rot90(k, (-2, -1))
                srf = sr.flip(-1)
                augm_state.append(sr)
                augm_state.append(srf)
            policy = self.forward(torch.cat(augm_state, dim=0))

            de_augm_policy = []
            for k in (0, 1, 2, 3):
                de_augm_policy.append(policy[k * 2][None].rot90(k, (-1, -2)))
                de_augm_policy.append(policy[k * 2 + 1][None].rot90(k, (-1, -2)).flip(-1))
            policy = torch.mean(torch.cat(de_augm_policy, dim=0), dim=0, keepdim=True)
        else:
            policy = self.forward(state)

        proba = self.model.predict_proba(state, policy)

        return policy[0][0], proba[0][0]

    def action(self, train=True, tta=False):
        eps = self.eps if train else 0

        policy, proba = self.forward_state(tta=tta)

        action, i, j, exploit = self._calc_move(policy=policy, eps=eps)
        value = self._forward_action(i, j)

        return policy, proba, action, exploit, value

    def ucb_action(self, q_est, u, q_est_skip_value=-2, lam=0.5, tta=False):
        policy, proba = self.forward_state(tta=tta)

        mask_est = q_est != q_est_skip_value
        policy[mask_est] = lam * policy[mask_est] + (1 - lam) * q_est[mask_est]
        policy = policy + proba * u

        action, i, j, exploit = self._calc_move(policy=policy, eps=0)
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
