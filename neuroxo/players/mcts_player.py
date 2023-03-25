from math import sqrt
from typing import Union

import numba as nb
import numpy as np
from dpipe.torch import to_device, to_np

from neuroxo.environment import Field
from neuroxo.im.augm import get_dihedral_augm_torch_fn
from neuroxo.numba.random import choose_idx, rand_argmax
from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.torch.utils import get_available_moves


class MCTSZeroPlayer:
    def __init__(self, model: NeuroXOZeroNN, field: Field, device: str = 'cpu', n_search_iter: int = 1600,
                 c_puct: float = 5, eps: float = 0.25, exploration_depth: int = 12, temperature: float = 1.,
                 reuse_tree: bool = True, deterministic_by_policy: bool = False, reduced_search: bool = True):
        self.model = to_device(model, device=device)
        self.field = field
        self.device = device

        self.n_search_iter = n_search_iter
        self.c_puct = c_puct
        self.eps = eps
        self.exploration_depth = exploration_depth
        self.temperature = temperature
        self.deterministic_by_policy = deterministic_by_policy
        self.reduced_search = reduced_search

        self.search_tree: Union[NodeState, None] = None
        self.reuse_tree = reuse_tree

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

    def _run_search(self):
        n, k, root_field = self.field.get_n(), self.field.get_k(), self.field.get_field()
        search_field = Field(n=n, k=k, field=root_field, device=self.device)

        if self.reuse_tree and (self.search_tree is not None):
            root = self.search_tree
            root.set_exploration_proba(eps=self.eps)
            n_start_iter = root.n if self.reduced_search else 0
        else:
            root = NodeState(None, search_field.get_running_features(), self.model, c_puct=self.c_puct, eps=self.eps)
            n_start_iter = 0

        for _ in range(n_start_iter, self.n_search_iter):
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

    @np.deprecate(message='We do not use resign value in this project and consider it deprecated.')
    def _get_resign_value(self):
        if self.search_tree is not None:
            value = self.search_tree.value
            if self.search_tree.expanded():
                a_max = max(self.search_tree.children, key=lambda a: self.search_tree.children[a].value)
                value = max(value, -self.search_tree.children[a_max].value)
            return value

    def _truncate_tree(self, a: int):
        if self.search_tree is not None:
            if a in self.search_tree.children:
                self.search_tree = self.search_tree.children[a]
                self.search_tree.parent = None
            else:
                self.search_tree = None

    @staticmethod
    @nb.njit()
    def _action(P, N, n, field_shape, exploration_move, t):
        if exploration_move:
            t = 1 / t
            pi = N ** t / n ** t  # np.power returns np.float64 in numba
            a = choose_idx(weights=pi)
        else:
            a = rand_argmax(N)
            pi = np.zeros_like(N, dtype=np.float64)  # therefore, we need the same type casting
            pi[a] = 1.

        pi = np.reshape(pi.astype(np.float32), field_shape)
        proba = np.reshape(P, field_shape)
        n_visits = np.reshape(N, field_shape)
        return a, pi, proba, n_visits

    def action(self):
        """ Returns (A, Pi, V, Proba, N_visits). """
        self._run_search()
        exploration_move = (not self.deterministic_by_policy) and (self.field.get_depth() <= self.exploration_depth)
        a, pi, proba, n_visits = self._action(self.search_tree.P, self.search_tree.N, self.search_tree.n,
                                              (self.n, self.n), exploration_move, self.temperature)
        v = self.search_tree.value

        self._make_move(*self.a2ij(a))
        if self.reuse_tree:
            self._truncate_tree(a)

        return a, pi, v, proba, n_visits

    def on_opponent_action(self, a: int):
        if self.reuse_tree:
            self._truncate_tree(a)

    def update_state(self, field):
        self.search_tree = None
        self.field.set_field(field)

    def set_field(self, field):
        self.search_tree = None
        self.field = field


class NodeState:
    def __init__(self, parent, running_features, model, c_puct: float = 5, eps: float = None):
        augm_fn, inv_augm_fn = get_dihedral_augm_torch_fn(return_inverse=True)
        proba, value = model(augm_fn(running_features))
        proba = inv_augm_fn(proba)

        self.avail_moves = to_np(get_available_moves(running_features)).ravel()

        self.value = value.item()
        self.P = to_np(proba).ravel()

        self.c_puct = c_puct
        self.eps = eps

        self.U = c_puct * self.P
        self.set_exploration_proba(eps=eps)

        self.N = np.zeros_like(self.P)
        self.n = 0
        self.Q = np.zeros_like(self.P)
        self.Q[self.avail_moves == 0] = -np.inf

        self.parent = parent
        self.last_action = None
        self.children = dict()

    @staticmethod
    @nb.njit()
    def _set_exploration_proba(p, am, eps: float = 0.25):
        p = p * (1 - eps) + eps * np.random.dirichlet(np.ones_like(p) * 0.02) * am
        return p / np.sum(p)

    def set_exploration_proba(self, eps: float = None):
        self.eps = eps
        if eps is not None:
            if eps > 0:
                p = self._set_exploration_proba(self.P, self.avail_moves, eps=eps)

                am = np.bool_(self.avail_moves)
                self.U[am] *= p[am] / self.P[am]
                self.P = p

    def expanded(self):
        return len(self.children) > 0

    def select(self):
        a = rand_argmax(self.Q + self.U)
        self.last_action = a
        return (a, self.children[a]) if (a in self.children) else (a, None)

    def expand(self, a, running_features, model, return_leaf: bool = False):
        self.children[a] = NodeState(self, running_features, model, c_puct=self.c_puct)
        if return_leaf:
            return self.children[a]

    def backup(self, value):
        a = self.last_action

        n_a = self.N[a]
        n_sum = self.n

        self.Q[a] *= n_a / (n_a + 1.)
        self.Q[a] += value / (n_a + 1.)

        self.U *= sqrt(n_sum + 1.) / max(1., sqrt(n_sum))
        self.U[a] *= (n_a + 1.) / (n_a + 2.)

        self.N[a] += 1
        self.n += 1

        if self.parent is not None:
            self.parent.backup(-value)
