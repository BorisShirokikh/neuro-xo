from typing import Union

import numpy as np

from neuroxo.environment.field import X_ID, Field
from neuroxo.im.augm import augm_spatial
from neuroxo.monte_carlo_tree_search import run_search, mcts_action
from neuroxo.players import PolicyPlayer, MCTSZeroPlayer


def play_self_game(player, field=None, train=True, augm=True, mcts=False, **mcts_kwargs):
    if field is None:
        field = np.zeros((player.field.get_n(), ) * 2, dtype='float32')
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
    v = player.field.get_value()  # value
    # v = None
    while v is None:
        s_history.append(player.field.get_field())
        f_history.append(player.field.get_features())

        if mcts:
            q, p, a, e, v = mcts_action(player=player, root=run_search(player=player, **mcts_kwargs))
        else:
            q, p, a, e, v = player.action(train=train)

        q_history.append(q)
        q_max_history.append(q[p > 0].max().item())
        p_history.append(p)
        a_history.append(a)
        e_history.append(e)

        if augm:
            player.update_field(augm_spatial(player.field.get_field()))

    return s_history, f_history, a_history, q_history, q_max_history, p_history, e_history, v


def play_duel(player_x: Union[PolicyPlayer, MCTSZeroPlayer],
              player_o: Union[PolicyPlayer, MCTSZeroPlayer],
              return_result_only: bool = False, self_game: bool = False, **action_kwargs):
    field = Field(n=player_x.n, kernel_len=player_x.field.get_kernel_len(), field=None, device=player_x.device)
    player_x.set_field(field=field)
    player_o.set_field(field=field)

    is_x_next = player_x.field.get_action_id() == X_ID

    player_act = player_x if is_x_next else player_o
    player_wait = player_o if is_x_next else player_x

    s_history = []      # states
    f_history = []      # features
    a_history = []      # actions
    o_history = []      # player.action outputs

    player_act.eval()
    if not self_game:
        player_wait.eval()
    v = field.get_value()  # value

    while v is None:
        s_history.append(field.get_field())
        f_history.append(field.get_features())

        a, *out = player_act.action(**action_kwargs)

        o_history.append(out)

        v = field.get_value()

        if not self_game:
            player_wait.on_opponent_action(a)

        if v is None:
            player_act, player_wait = player_wait, player_act

    if v == 0:
        winner = 0
    else:  # v == 1:
        winner = player_act.field.get_opponent_action_id()

    return winner if return_result_only else (s_history, f_history, a_history, o_history, winner)
