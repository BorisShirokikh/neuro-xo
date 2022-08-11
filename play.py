import argparse
import sys

import numpy as np
import pygame
from pygame import SYSTEM_CURSOR_ARROW, SYSTEM_CURSOR_WAIT
from pygame.cursors import Cursor
from dpipe.torch import load_model_state
from dpipe.io import load_json

from gui.board import Board
from neuroxo.environment.field import Field, O_ID, X_ID
from neuroxo.players import MCTSZeroPlayer
from neuroxo.torch.module.zero_net import NeuroXOZeroNN
from neuroxo.utils import get_repo_root, flush


def terminate():
    pygame.display.quit()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe 10x10")

    # ### load agent ###
    agent_name = 'zero_v0'
    agent_path = get_repo_root() / 'agents' / agent_name

    config = load_json(agent_path / 'config.json')
    n, k = config['n'], config['k']
    device = 'cpu'

    in_channels, n_blocks, n_features = config['in_channels'], config['n_blocks'], config['n_features']
    c_puct = config['c_puct']
    reuse_tree = config['reuse_tree']

    # n_search_iter = config['n_search_iter']
    # eps = config['eps']
    exploration_depth = config['exploration_depth']
    temperature = config['temperature']
    # deterministic_by_policy = config['deterministic_by_policy']
    # reduced_search = config['reduced_search']
    n_search_iter = 1200
    eps = 0
    deterministic_by_policy = True
    reduced_search = False

    field = Field(n=n, k=k, field=None, device=device)
    model = NeuroXOZeroNN(n, in_channels=in_channels, n_blocks=n_blocks, n_features=n_features)

    player = MCTSZeroPlayer(model=model, field=field, device=device, n_search_iter=n_search_iter,
                            c_puct=c_puct, eps=eps, exploration_depth=exploration_depth, temperature=temperature,
                            reuse_tree=reuse_tree,
                            deterministic_by_policy=deterministic_by_policy, reduced_search=reduced_search)

    load_model_state(player.model, agent_path / 'model.pth')
    # ### ########## ###

    player_1, player_2 = np.random.choice([X_ID, O_ID], size=2, replace=False)
    game_over, winner = False, None

    # ###

    prev_move = None
    cur_player = X_ID

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
                    a, *out = player.action()
                    flush(f'>>> Estimated position value: {out[2]:.2f}')
                    v = player.field.get_value()
                    pygame.mouse.set_cursor(Cursor(SYSTEM_CURSOR_ARROW))

                    clicked_row, clicked_col = a // n, a % n
                    move = clicked_row, clicked_col

                    board.field = player.field.get_field()

                    board.color_prev(screen, *move)

                    if prev_move:
                        original_color = board.field_colors[(prev_move[0] % 2 + prev_move[1] % 2) % 2]
                        board.color_prev(screen, *prev_move, color=original_color)

                    prev_move = move
                    cur_player = player.field.get_action_id()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouseX = event.pos[0]
                    mouseY = event.pos[1]

                    clicked_row = int(np.clip(int((mouseY - board.frame_thickness) // board.cell_size), 0, n - 1))
                    clicked_col = int(np.clip(int((mouseX - board.frame_thickness) // board.cell_size), 0, n - 1))

                    move = clicked_row, clicked_col

                    if board.field[clicked_row, clicked_col] == 0:
                        player.field.make_move(clicked_row, clicked_col)
                        a = player.ij2a(*move)
                        v = player.field.get_value()
                        if args.mode == 'single':
                            player.on_opponent_action(a)

                        board.field = player.field.get_field()

                        board.color_prev(screen, *move)

                        if prev_move:
                            original_color = board.field_colors[(prev_move[0] % 2 + prev_move[1] % 2) % 2]
                            board.color_prev(screen, *prev_move, color=original_color)

                        prev_move = move
                        cur_player = player.field.get_action_id()

            board.draw_position(screen)

            is_win, by, at = field.check_win(return_how=True)
            is_draw = field.check_draw()

            if is_win:
                board.color_win(screen, by, at, k)

            if is_win or is_draw:
                game_over = True

            if event.type == pygame.QUIT:
                terminate()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    terminate()
                elif event.key == pygame.K_r and game_over:
                    player.update_state(np.zeros((n, n), dtype='float32'))
                    board.field = player.field.get_field()
                    board.clear_board(screen)

                    player_1, player_2 = player_2, player_1
                    game_over = False
