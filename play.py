import argparse

import sys
import time
from pathlib import Path

import pygame

import numpy as np
from dpipe.torch import load_model_state

from gui.board import Board
from ttt_lib.field import Field
from ttt_lib.monte_carlo_tree_search import mcts_action, run_search
from ttt_lib.policy_player import PolicyPlayer
from ttt_lib.torch.module.policy_net import PolicyNetworkQ10Light


def terminate():
    pygame.display.quit()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe 10x10")

    # ###

    n, kernel_len = 10, 5
    device = 'cpu'

    search_time = 5

    cnn_features = (128, 64)

    field = Field(n=n, kernel_len=kernel_len, device=device, check_device=device)

    model = PolicyNetworkQ10Light(n=n, structure=cnn_features)
    model_path = Path(__file__).absolute().parent / 'agents' / 'q_10x10' / 'model_5.pth'
    load_model_state(model, model_path)

    eps = 0
    player = PolicyPlayer(model=model, field=field, eps=eps, device=device)
    player.eval()

    player_1, player_2 = np.random.choice([1, -1], size=2, replace=False)
    game_over, winner = False, None

    # ###

    last_move = None
    cur_player = player_1

    field_size, frame_thickness = 50 * n, 10

    screen = pygame.display.set_mode((field_size + 2 * frame_thickness,) * 2, )

    board = Board(screen, n, 50 * n, frame_thickness)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['single', 'pvp'])
    args = parser.parse_args()

    while True:
        pygame.display.update()

        for event in pygame.event.get():
            if not game_over:
                if args.mode == 'single' and player.field.get_action_id() == player_2:
                    _, _, a, v, _ = mcts_action(player=player, root=run_search(player=player, search_time=search_time))

                    clicked_row, clicked_col = a // n, a % n
                    move = clicked_row, clicked_col

                    board.field[clicked_row, clicked_col] = cur_player

                    board.color_cell(screen, *move)

                    if last_move:
                        original_color = board.field_colors[(last_move[0] % 2 + last_move[1] % 2) % 2]
                        board.color_cell(screen, *last_move, color=original_color)

                    last_move = move
                    cur_player *= -1

                    # q, p, a, v, e = player.action(train=False)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]

                    clicked_row = int(np.clip(int((mouseY - board.frame_thickness) // board.cell_size), 0, n - 1))
                    clicked_col = int(np.clip(int((mouseX - board.frame_thickness) // board.cell_size), 0, n - 1))

                    move = clicked_row, clicked_col

                    if board.field[clicked_row, clicked_col] == 0:
                        _, _, a, v, _ = player.manual_action(clicked_row, clicked_col)

                        board.field[clicked_row, clicked_col] = cur_player

                        board.color_cell(screen, *move)

                        if last_move:
                            original_color = board.field_colors[(last_move[0] % 2 + last_move[1] % 2) % 2]
                            board.color_cell(screen, *last_move, color=original_color)

                        last_move = move
                        cur_player *= -1

                    time.sleep(.2)

            is_win, by, at = field.check_win(return_how=True)
            is_draw = field.check_draw()

            if is_win or is_draw:
                game_over = True

            if (event.type == pygame.KEYDOWN and event.key == pygame.K_q) or (event.type == pygame.QUIT):
                terminate()

        board.draw_position(screen)
