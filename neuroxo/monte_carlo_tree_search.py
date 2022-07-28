import time
from collections import defaultdict
from copy import deepcopy

import numpy as np
from dpipe.torch import to_var

from neuroxo.environment.field import Field
from neuroxo.policy_player import PolicyPlayer
from neuroxo.utils import make_state_hashable, get_hash_table


def calc_alpha_go_ucb(tree, h, n, c=5, n_key='n', sum_key='sum', q_est_skip_value=-2):
    q_est = np.zeros((n, n), dtype='float32') + q_est_skip_value
    u_denominator = np.ones((n, n), dtype='float32')
    for a, n_sum in tree[h].items():
        i, j = a // n, a % n
        n_visits = n_sum[n_key]
        u_denominator[i, j] += n_visits
        q_est[i, j] += (n_sum[sum_key] / n_visits)
    u = c * np.sqrt((np.sum(u_denominator) - n ** 2) / u_denominator)
    return q_est, u


def calc_classical_ucb(tree, h, n, c=5, n_key='n'):
    # TODO: not implemented
    u_denominator = np.ones((n, n), dtype='float32')
    for a, n_sum in tree[h].items():
        u_denominator[a // n, a % n] += n_sum[n_key]
    u = c * np.sqrt(np.sum(u_denominator) - n ** 2)
    return u


def play_search_game(player, tree, n, field=None):
    if field is None:
        field = np.zeros((player.field.get_n(), ) * 2, dtype='float32')
    player.update_field(field=field)

    s_history = []      # state hashes
    a_history = []      # actions
    q_history = []      # max Q values
    p_history = []      # max P values

    player.eval()
    v = None  # v -- value
    while v is None:
        h = hash(make_state_hashable(player.field.get_field()))
        q_est, u = calc_alpha_go_ucb(tree=tree, h=h, n=n)
        q_est, u = to_var(q_est, device=player.device), to_var(u, device=player.device)
        q, p, a, _, v = player.ucb_action(q_est, u)

        s_history.append(h)
        p_history.append(p.max().item())
        q_history.append(q[p > 0].max().item())
        a_history.append(a)

    return s_history, a_history, q_history, p_history, v


def run_search(player: PolicyPlayer, search_time=60, n_key='n', sum_key='sum', return_tree=False):
    n = player.field.get_n()
    root_field = player.field.get_field()

    search_field = Field(n=player.field.get_n(), kernel_len=player.field.kernel_len,
                         field=player.field.get_field(), device=player.device, check_device=player.device)
    search_player = PolicyPlayer(model=deepcopy(player.model), field=search_field, device=player.device)

    tree = get_hash_table()
    root_table = defaultdict(lambda: defaultdict(float))

    start_time = time.time()
    while (time.time() - start_time) < search_time:
        s_history, a_history, q_history, p_history, v = play_search_game(player=search_player, tree=tree, n=n,
                                                                         field=root_field)

        value = v
        for i, (s, a) in enumerate(zip(s_history[::-1], a_history[::-1])):
            if i == len(a_history) - 1:
                root_table[a][n_key] += 1
                root_table[a][sum_key] += value
            tree[s][a][n_key] += 1
            tree[s][a][sum_key] += value
            value = -value

    return (root_table, tree) if return_tree else root_table


def mcts_action(player: PolicyPlayer, root, n_key='n', sum_key='sum'):
    n = player.n
    q = np.zeros((n, n), dtype='float32') - 1
    n_visits_instead_of_p = np.zeros((n, n), dtype='float32')
    for a, n_sum in root.items():
        q[a // n, a % n] = n_sum[sum_key] / n_sum[n_key]
        n_visits_instead_of_p[a // n, a % n] = n_sum[n_key]
    a = np.random.choice((n_visits_instead_of_p == n_visits_instead_of_p.max()).ravel().nonzero()[0])
    _, _, _, e, v = player.manual_action(a // n, a % n)
    return q, n_visits_instead_of_p, a, e, v
