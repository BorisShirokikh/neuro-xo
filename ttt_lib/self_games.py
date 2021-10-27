import numpy as np

from ttt_lib.im.augm import augm_spatial
from ttt_lib.monte_carlo_tree_search import run_search, mcts_action


def play_self_game(player, field=None, train=True, augm=True, mcts=False, **mcts_kwargs):
    if field is None:
        field = np.zeros((player.field.get_size(), ) * 2, dtype='float32')
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
        s_history.append(player.field.get_state())
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
            player.update_field(augm_spatial(player.field.get_state()))

    return s_history, f_history, a_history, q_history, q_max_history, p_history, e_history, v


def play_duel(player_x, player_o, field=None, return_result_only=False, tta_x=False, tta_o=False,
              mcts_x=False, mcts_o=False, search_time_x=10, search_time_o=10):
    if field is None:
        field = np.zeros((player_x.field.get_size(), ) * 2, dtype='float32')
    player_x.update_field(field=field)

    is_x_next = player_x.field.next_action_id == 1

    player_act = player_x if is_x_next else player_o
    player_wait = player_o if is_x_next else player_x

    tta_act = tta_x if is_x_next else tta_o
    tta_wait = tta_o if is_x_next else tta_x

    mcts_act = mcts_x if is_x_next else mcts_o
    mcts_wait = mcts_o if is_x_next else mcts_x

    search_time_act = search_time_x if is_x_next else search_time_o
    search_time_wait = search_time_o if is_x_next else search_time_x

    s_history = []  # states
    f_history = []  # features
    a_history = []  # actions
    e_history = []  # exploitations
    q_history = []  # policies
    p_history = []  # probas

    player_act.eval()
    player_wait.eval()
    v = player.field.get_value()  # value
    while v is None:
        s_history.append(player_act.field.get_state())
        f_history.append(player_act.field.get_features())

        if mcts_act:
            q, p, a, e, v = mcts_action(player=player_act,
                                        root=run_search(player=player_act, search_time=search_time_act))
        else:
            q, p, a, e, v = player_act.action(train=True, tta=tta_act)

        q_history.append(q)
        p_history.append(p)
        a_history.append(a)
        e_history.append(e)

        if v is None:
            player_act, player_wait = player_wait, player_act
            tta_act, tta_wait = tta_wait, tta_act
            mcts_act, mcts_wait = mcts_wait, mcts_act
            search_time_act, search_time_wait = search_time_wait, search_time_act

    if v == 0:
        winner = 0
    else:  # v == 1:
        winner = - player_act.field.next_action_id

    return winner if return_result_only else (s_history, f_history, a_history, q_history, p_history, e_history, winner)
