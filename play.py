import sys
import pygame

import numpy as np

from gui.board import Board


def terminate():
    pygame.display.quit()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Tic-Tac-Toe 10x10")

    n = 10

    last_move = None
    player = 1

    field_size, frame_thickness = 50 * n, 10

    screen = pygame.display.set_mode((field_size + 2 * frame_thickness,) * 2, )

    board = Board(screen, n, 50 * n, frame_thickness)

    while True:
        pygame.display.update()

        for event in pygame.event.get():
            if (event.type == pygame.KEYDOWN and event.key == pygame.K_q) or (event.type == pygame.QUIT):
                terminate()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouseX = event.pos[0]
                mouseY = event.pos[1]

                clicked_row = int(np.clip(int((mouseY - board.frame_thickness) // board.cell_size), 0, n - 1))
                clicked_col = int(np.clip(int((mouseX - board.frame_thickness) // board.cell_size), 0, n - 1))

                move = clicked_row, clicked_col

                if board.field[clicked_row, clicked_col] == 0:
                    board.field[clicked_row, clicked_col] = player

                    board.color_cell(screen, *move)

                    if last_move:
                        original_color = board.field_colors[(last_move[0] % 2 + last_move[1] % 2) % 2]
                        board.color_cell(screen, *last_move, color=original_color)

                    last_move = move
                    player *= -1

        board.draw_position(screen)
