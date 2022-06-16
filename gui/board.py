import pygame

import numpy as np


class Board:
    def __init__(self, screen, n, field_size, frame_thickness):
        self.n = n

        self.field = np.zeros((n, n))

        self.field_size = field_size
        self.board_size = field_size + 2 * frame_thickness

        self.frame_thickness = frame_thickness
        self.edge_thickness = 1

        self.cell_size = self.field_size // self.n

        self.field_colors = [(118, 150, 86), (238, 238, 210)]

        self._fill_bg(screen)
        self._color_field(screen, self.field_colors[0], self.field_colors[1])
        self._draw_frame(screen)

        self.cross = pygame.image.load('../assets/cross_h.png')
        self.circle = pygame.image.load('../assets/circle_h.png')

        self.prev_move = None

    def _fill_bg(self, screen, color: tuple = (51, 46, 43)):
        screen.fill(color)

    def _draw_frame(self, screen, color: tuple = (0, 0, 0)):
        start, stop = (0,) * 2, (self.board_size,) * 2
        rect = pygame.Rect(*start, *stop)

        pygame.draw.rect(
            screen,
            color=color,
            rect=rect,
            width=self.frame_thickness
        )

    def _draw_edges(self, screen, color: tuple = (0, 0, 0)):
        def hedge(pos):
            return pygame.draw.line(
                screen,
                color=color,
                start_pos=(self.frame_thickness, self.frame_thickness + i * self.cell_size),
                end_pos=(self.board_size - self.frame_thickness, self.frame_thickness + i * self.cell_size),
                width=self.edge_thickness
            )

        def wedge(pos):
            return pygame.draw.line(
                screen,
                color=color,
                start_pos=(self.frame_thickness + i * self.cell_size, self.frame_thickness),
                end_pos=(self.frame_thickness + i * self.cell_size, self.board_size - self.frame_thickness),
                width=self.edge_thickness
            )

        for i in range(1, self.n):
            hedge(i)
            wedge(i)

    def _color_field(self, screen, cell_color: tuple, field_color: tuple):
        start, stop = (self.frame_thickness,) * 2, (self.field_size,) * 2
        rect = pygame.Rect(*start, *stop)

        pygame.draw.rect(
            surface=screen,
            color=field_color,
            rect=rect
        )

        for i in range(0, self.n):
            for j in range(i % 2, self.n + i % 2, 2):
                start = (self.frame_thickness + i * self.cell_size, self.frame_thickness + j * self.cell_size)
                stop = (self.cell_size,) * 2

                pygame.draw.rect(
                    surface=screen,
                    color=cell_color,
                    rect=pygame.Rect(*start, *stop)
                )

    def draw_position(self, screen, offset: float = .1):
        for row in range(self.n):
            for col in range(self.n):
                if self.field[row][col] == 1:
                    obj = self.cross
                elif self.field[row][col] == -1:
                    obj = self.circle
                else:
                    continue

                start = np.array([col, row]) * self.cell_size + (self.frame_thickness + offset * self.cell_size // 2)

                obj = pygame.transform.scale(obj, (self.cell_size * (1 - offset),) * 2)
                screen.blit(obj, start)

    def color_cell(self, screen, row: int, col: int, color: tuple = (186, 202, 69)):
        start = (self.frame_thickness + col * self.cell_size, self.frame_thickness + row * self.cell_size)
        stop = (self.cell_size,) * 2

        pygame.draw.rect(
            surface=screen,
            color=color,
            rect=pygame.Rect(*start, *stop)
        )
