import sys
from pathlib import Path

import pygame

import numpy as np
from dpipe.io import choose_existing

from dpipe.torch import load_model_state

from alphaxo.field import Field, WINDOW, BG_COLOR
from alphaxo.monte_carlo_tree_search import mcts_action, run_search
from alphaxo.torch.module.policy_net import PolicyNetworkQ10Light
from alphaxo.q_learning import PolicyPlayer
from alphaxo.utils import choose_model

Q_EXP_PATH = choose_existing(
    Path('/nmnt/x4-hdd/experiments/rl/q_10x10/'),
    Path.cwd() / '../agents/q_10x10/'
)

if __name__ == '__main__':
    print('\n')
    print('Tic-Tac-Toe Game\n')
    print('Commands:')
    print('\t R - restart the game')
    print('\t Q - quit the game\n')

    available_exp_names = ' ,'.join([x.name for x in Q_EXP_PATH.glob("*")])

    print(f'Choose the opponent: {available_exp_names}')
    exp_name = input()
    while exp_name not in available_exp_names:
        print(f'There is no {exp_name} model. Please choose from the available.')
        exp_name = input()

    print(f'Input the search time (int):')
    search_time = int(input())

    n, kernel_len = 10, 5
    device = 'cpu'

    cnn_features = (128, 64)

    screen = pygame.display.set_mode((WINDOW, WINDOW))
    screen.fill(BG_COLOR)

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)
    field.draw_field(screen)

    model = PolicyNetworkQ10Light(n=n, structure=cnn_features)
    model_path = Q_EXP_PATH / exp_name
    load_model_state(model, model_path / choose_model(model_path))

    a = None
    eps = 0
    player = PolicyPlayer(model=model, field=field, eps=eps, device=device)
    player.eval()

    player_1, player_2 = np.random.choice([1, -1], size=2, replace=False)
    game_over, winner = False, None

    while True:
        for event in pygame.event.get():
            if not game_over:
                if player.field.next_action_id == player_1:
                    _, _, a, v, _ = mcts_action(player=player, root=run_search(player=player, search_time=search_time))
                    # q, p, a, v, e = player.action(train=False)
                else:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        mouseX = event.pos[0]
                        mouseY = event.pos[1]

                        clicked_row = int(mouseY // field.square_size)
                        clicked_col = int(mouseX // field.square_size)

                        if field.get_state()[clicked_row, clicked_col] == 0:
                            _, _, a, v, _ = player.manual_action(clicked_row, clicked_col)

            field.draw_figures(screen)

            is_win, by, at = field.check_win(return_how=True)
            is_draw = field.check_draw()

            if is_win:
                game_over = True
                last_move = a // n, a % n
                field.draw_winning_line(screen, by, at)
            elif is_draw:
                game_over = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    screen.fill(BG_COLOR)
                    field.draw_field(screen)
                    player.update_field(np.zeros((n, n), dtype='float32'))

                    player_1, player_2 = player_2, player_1
                    game_over = False
                elif event.key == pygame.K_q:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
