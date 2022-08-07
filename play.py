import argparse
import sys

import numpy as np
import pygame
from dpipe.torch import load_model_state
from pygame import SYSTEM_CURSOR_ARROW, SYSTEM_CURSOR_WAIT
from pygame.cursors import Cursor

from gui.board import Board
from neuroxo.environment.field import Field
from neuroxo.monte_carlo_tree_search import mcts_action, run_search
from neuroxo.policy_player import PolicyPlayer
from neuroxo.torch.module.policy_net import PolicyNetworkQ10Light
from neuroxo.utils import get_repo_root


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

    field = Field(n=n, k=kernel_len, device=device, check_device=device)

    model = PolicyNetworkQ10Light(n=n, structure=cnn_features)
    model_path = get_repo_root() / 'agents' / 'q_10x10' / 'model_5.pth'
    load_model_state(model, model_path)

    eps = 0
    player = PolicyPlayer(model=model, field=field, eps=eps, device=device)
    player.eval()

    player_1, player_2 = np.random.choice([1, -1], size=2, replace=False)
    game_over, winner = False, None

    # ###

    prev_move = None
    cur_player = 1

    field_size, frame_thickness = 50 * n, 10

    screen = pygame.display.set_mode((field_size + 2 * frame_thickness,) * 2, )

    # TODO: embed Board in Field
    board = Board(screen, n, 50 * n, frame_thickness)

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, choices=['single', 'pvp'])
    args = parser.parse_args()

    while True:
        for event in pygame.event.get():
            if not game_over:
                if args.mode == 'single' and player.field.get_action_id() == player_2:
                    pygame.mouse.set_cursor(Cursor(SYSTEM_CURSOR_WAIT))
                    _, _, a, v, _ = mcts_action(player=player, root=run_search(player=player, search_time=search_time))
                    pygame.mouse.set_cursor(Cursor(SYSTEM_CURSOR_ARROW))

                    clicked_row, clicked_col = a // n, a % n
                    move = clicked_row, clicked_col

                    board.field = player.field.get_field()

                    board.color_prev(screen, *move)

                    if prev_move:
                        original_color = board.field_colors[(prev_move[0] % 2 + prev_move[1] % 2) % 2]
                        board.color_prev(screen, *prev_move, color=original_color)

                    prev_move = move
                    cur_player *= -1

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]

                    clicked_row = int(np.clip(int((mouseY - board.frame_thickness) // board.cell_size), 0, n - 1))
                    clicked_col = int(np.clip(int((mouseX - board.frame_thickness) // board.cell_size), 0, n - 1))

                    move = clicked_row, clicked_col

                    if board.field[clicked_row, clicked_col] == 0:
                        _, _, a, v, _ = player.manual_action(clicked_row, clicked_col)

                        board.field = player.field.get_field()

                        board.color_prev(screen, *move)

                        if prev_move:
                            original_color = board.field_colors[(prev_move[0] % 2 + prev_move[1] % 2) % 2]
                            board.color_prev(screen, *prev_move, color=original_color)

                        prev_move = move
                        cur_player *= -1

            board.draw_position(screen)

            is_win, by, at = field.check_win(return_how=True)
            is_draw = field.check_draw()

            if is_win:
                board.color_win(screen, by, at, kernel_len)

            if is_win or is_draw:
                game_over = True

            if event.type == pygame.QUIT:
                terminate()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminate()
                elif event.key == pygame.K_r and game_over:
                    player.update_field(np.zeros((n, n), dtype='float32'))
                    board.field = player.field.get_field()
                    board.clear_board(screen)

                    player_1, player_2 = player_2, player_1
                    game_over = False
