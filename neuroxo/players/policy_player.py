import numpy as np

from dpipe.torch import to_device, to_np

from neuroxo.torch.module.policy_net import ProbaPolicyNN
from neuroxo.environment import Field


class PolicyPlayer:
    def __init__(self, model: ProbaPolicyNN, field: Field, device: str = 'cpu', eps: float = 0.25):
        self.model = to_device(model, device=device)
        self.field = field
        self.device = device
        self.eps = eps

        self.n = self.field.get_n()

        self.eval()

    def _make_move(self, i, j):
        self.field.make_move(i, j)

    def _calc_move(self, proba: np.ndarray):
        """
        eps = None  : proba choice          : on-policy
        eps = 0     : greedy choice         : greedy
        eps > 0     : eps-greedy choice     : off-policy
        """
        if self.eps is None:
            p = proba.ravel()
            return np.random.choice(np.arange(len(p)), p=p)
        else:
            if self.eps == 0:
                return proba.argmax()
            else:
                n = self.n ** 2
                p = proba.ravel()
                unavail_a = self.field._field.ravel().nonzero()

                soft_p = np.zeros((n, ), dtype=np.float32) + self.eps / (n - len(unavail_a[0]))
                soft_p[unavail_a] = 0
                soft_p[p.argmax()] += (1 - self.eps)

                return np.random.choice(np.arange(n), p=soft_p)

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

    def action(self):
        proba, policy = self.forward(self.field.get_running_features())
        proba, policy = to_np(proba[0][0]), to_np(policy[0][0])
        action = self._calc_move(proba)

        self._make_move(*self.a2ij(action))

        return action, proba, policy

    def on_opponent_action(self, a: int):
        pass

    def update_state(self, field):
        self.field.set_field(field)

    def set_field(self, field):
        self.field = field
