from math import sqrt
from typing import Sequence, Union

import numpy as np
import torch.nn as nn
from dpipe.torch import to_device, to_np

from neuroxo.environment import Field


class MCTSZeroPlayer:
    def __init__(self, model: nn.Module, field: Field, device: str = 'cpu',
                 n_search_iter: int = 1600, c_puct: float = 5, eps: float = 0.25, exploration_depth: int = 10,
                 temperature: float = 1., reuse_tree: bool = False):
        self.model = to_device(model, device=device)
        self.field = field
        self.device = device

        self.n_search_iter = n_search_iter
        self.c_puct = c_puct
        self.eps = eps
        self.exploration_depth = exploration_depth
        self.temperature = temperature

        self.search_tree: Union[NodeState, None] = None
        self.reuse_tree = reuse_tree

        # fields to slightly speedup computation:
        self.n = self.field.get_n()

        self.eval()

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

    def _make_move(self, i, j):
        self.field.make_move(i, j)
        return self.field.get_value()

    def _run_search(self):
        n, k, root_field = self.field.get_n(), self.field.get_kernel_len(), self.field.get_field()
        search_field = Field(n=n, kernel_len=k, field=root_field, device=self.device)

        if self.reuse_tree and (self.search_tree is not None):
            root = self.search_tree
        else:
            root = NodeState(None, search_field.get_running_features(), self.model, c_puct=self.c_puct, eps=self.eps)

        for _ in range(self.n_search_iter):
            node = root
            parent = None

            while node is not None:
                parent = node
                action, node = node.select()
                search_field.make_move(*self.a2ij(action))

            value = search_field.get_value()

            if value is None:
                value = - parent.expand(action, search_field.get_running_features(), self.model, return_leaf=True).value

            parent.backup(value)
            search_field.set_field(root_field)

        self.search_tree = root

    def _get_resign_value(self):
        if self.search_tree is not None:
            value = self.search_tree.value
            if self.search_tree.expanded():
                a_max = max(self.search_tree.children, key=lambda a: self.search_tree.children[a].value)
                value = max(value, -self.search_tree.children[a_max].value)
            return value

    def _truncate_tree(self, a: int):
        if self.search_tree is not None:
            if a in list(self.search_tree.children.keys()):
                self.search_tree = self.search_tree.children[a]
            else:
                self.search_tree = None

    def action(self, deterministic_by_policy: bool = False):
        """ Returns (A, P, V, V_resign). """
        self._run_search()
        N, n = self.search_tree.N, self.search_tree.n

        if (not deterministic_by_policy) and (self.field.get_depth() <= self.exploration_depth):
            p = N ** (1 / self.temperature) / n ** (1 / self.temperature)
            a = np.random.choice(np.arange(len(N)), p=p)
        else:
            a = np.argmax(N)
            p = np.zeros_like(N, dtype=np.float32)
            p[a] = 1

        p, v, v_resign = np.reshape(p, (self.n, self.n)), self.search_tree.value, self._get_resign_value()

        self._make_move(*self.a2ij(a))

        if self.reuse_tree:
            self._truncate_tree(a)

        return a, p, v, v_resign

    def on_opponent_action(self, a: int):
        if self.reuse_tree:
            self._truncate_tree(a)

    def update_state(self, field):
        self.field.set_field(field)

    def set_field(self, field):
        self.field = field


class NodeState:
    def __init__(self, parent, running_features, model, c_puct: float = 5, eps: float = None):
        proba, value = model(running_features)

        self.value = value.item()

        self.P = to_np(proba[0][0]).ravel()

        self.eps = eps
        if eps is not None:
            self.P *= (1 - eps)
            self.P += eps * np.random.dirichlet(np.ones_like(self.P) * 0.02) * np.float32(self.P > 0)
            self.P /= self.P.sum()

        self.N = np.zeros_like(self.P)
        self.n = 0
        self.Q = np.zeros_like(self.P)

        self.c = c_puct
        self.U = c_puct * self.P

        self.parent = parent
        self.last_action = None
        self.children = dict()

    def expanded(self):
        return len(self.children) > 0

    def select(self):
        a = np.argmax(self.Q + self.U)
        self.last_action = a
        return (a, self.children[a]) if a in list(self.children.keys()) else (a, None)

    def expand(self, a, running_features, model, return_leaf=False):
        self.children[a] = NodeState(self, running_features, model, c_puct=self.c)
        if return_leaf:
            return self.children[a]

    def backup(self, value):
        a = self.last_action

        n_a = self.N[a]
        n_sum = self.n

        self.Q[a] *= n_a / (n_a + 1.)
        self.Q[a] += value / (n_a + 1.)

        self.U *= max(1., sqrt(n_sum)) / sqrt(n_sum + 1.)
        self.U[a] *= (n_a + 2.) / (n_a + 1.)

        self.N[a] += 1
        self.n += 1

        if self.parent is not None:
            self.parent.backup(-value)
