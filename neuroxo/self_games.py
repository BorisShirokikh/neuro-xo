from neuroxo.environment.field import X_ID, Field
from neuroxo.players import MCTSZeroPlayer


def play_duel(player_x: MCTSZeroPlayer, player_o: MCTSZeroPlayer,
              return_result_only: bool = False, self_game: bool = False):
    field = Field(n=player_x.n, k=player_x.field.get_k(), field=None, device=player_x.device)
    player_x.set_field(field=field)
    player_o.set_field(field=field)

    is_x_next = player_x.field.get_action_id() == X_ID

    player_act = player_x if is_x_next else player_o
    player_wait = player_o if is_x_next else player_x

    s_history = []      # states
    f_history = []      # features
    a_history = []      # actions
    o_history = []      # player.action outputs

    v = field.get_value()  # value

    while v is None:
        if not return_result_only:
            s_history.append(field.get_field())
            f_history.append(field.get_features())

        a, *out = player_act.action()

        if not return_result_only:
            a_history.append(a)
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
