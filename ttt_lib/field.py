import torch
import torch.nn as nn
import numpy as np
import pygame
import matplotlib.pyplot as plt
from dpipe.torch import to_np, to_device, to_var


DIRECTIONS = ('row', 'col', 'diag', 'diag1')

WIN_VALUE = 1
DRAW_VALUE = 0

WINDOW = 500
BG_COLOR = (255, 255, 255)
LINE_COLOR = CIRCLE_COLOR = CROSS_COLOR = (0, 0, 0)
WIN_LINE_COLOR = (255, 0, 0)


def _check_numpy_field(field: np.ndarray):
    if len(field.shape) != 2:
        raise ValueError(f'Field could be initialized only with 2-dim arrays; '
                         f'however, {len(field.shape)}-dim array is given')
    if field.shape[0] != field.shape[1]:
        raise ValueError(f'Working with square fields only; however, shape `{field.shape}` is given.')


def get_check_kernel(kernel_len=5, direction='row'):
    if direction not in DIRECTIONS:
        raise ValueError(f'`direction` must be one of the {DIRECTIONS}, {direction} given')

    if 'diag' in direction:
        kernel_w = torch.tensor(np.array([[np.eye(kernel_len, dtype=np.float32)]]))
        kernel = nn.Conv2d(1, 1, kernel_len, bias=False)
        if '1' in direction:
            kernel.weight = nn.Parameter(kernel_w.flip(dims=(-1,)), requires_grad=False)
        else:
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        kernel.eval()
        return kernel
    else:
        kernel_w = torch.tensor([[[[1.] * kernel_len]]])
        if 'row' in direction:
            kernel = nn.Conv2d(1, 1, (1, kernel_len), bias=False)
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        else:
            kernel = nn.Conv2d(1, 1, (kernel_len, 1), bias=False)
            kernel.weight = nn.Parameter(kernel_w.transpose(-2, -1), requires_grad=False)
        kernel.eval()
        return kernel


class Field:
    def __init__(self, n=10, kernel_len=5, field=None, device='cpu', check_device='cpu'):
        if field is not None:
            if not isinstance(field, np.ndarray):
                field = to_np(field)
            _check_numpy_field(field)
            self._field = np.copy(np.float32(field))
            self._n = self._field.shape[0]
        else:
            self._n = n
            self._field = np.zeros((n, n), dtype='float32')

        self.kernel_len = kernel_len
        self.next_action_id = 1

        self.device = device
        self._features = self.field2features()
        self._running_features = to_var(np.copy(self._features), device=self.device)
        self.n_features = self._features.shape[1]

        self.check_device = check_device
        self._check_field = to_var(self._field[None, None], device=check_device)
        self._row_kernel = to_device(get_check_kernel(kernel_len=kernel_len, direction='row'), device=check_device)
        self._col_kernel = to_device(get_check_kernel(kernel_len=kernel_len, direction='col'), device=check_device)
        self._diag_kernel = to_device(get_check_kernel(kernel_len=kernel_len, direction='diag'), device=check_device)
        self._diag1_kernel = to_device(get_check_kernel(kernel_len=kernel_len, direction='diag1'), device=check_device)

        self.square_size = WINDOW // self._n

    def _update_action_id(self):
        self.next_action_id = -self.next_action_id
        self._features = self._features[:, [1, 0, *range(2, self.n_features)], ...]
        self._running_features = self._running_features[:, [1, 0, *range(2, self.n_features)], ...]

    def field2features(self, field=None):
        field = self._field if field is None else np.copy(field)
        next_action_id = 1 if (self.get_depth(field) % 2 == 0) else -1
        features = np.float32(np.concatenate((
            (field == next_action_id)[None, None],  # is player?      {0, 1}
            (field == -next_action_id)[None, None],  # is opponent?    {0, 1}
            np.zeros_like(field)[None, None],  # zeros plane     {0}
            np.ones_like(field)[None, None],  # ones plane      {1}
            (field == 0)[None, None],  # is legal move?  {0, 1}
        ), axis=1))
        return features

    def make_move(self, i, j, _id=None):
        if _id is None:
            _id = self.next_action_id

        if self._field[i, j] != 0:
            raise IndexError('Accessing already marked cell.')

        self._field[i, j] = _id
        self._check_field[0, 0, i, j] = _id

        self._features[0, 0, i, j] = 1
        self._features[0, -1, i, j] = 0

        self._running_features[0, 0, i, j] = 1
        self._running_features[0, -1, i, j] = 0

        self._update_action_id()

    def backward_move(self, i, j):
        if self._field[i, j] == 0:
            raise IndexError('Clearing already empty cell.')

        self._field[i, j] = 0
        self._check_field[0, 0, i, j] = 0

        self._features[0, 0, i, j] = 0
        self._features[0, -1, i, j] = 1

        self._running_features[0, 0, i, j] = 0
        self._running_features[0, -1, i, j] = 1

        self._update_action_id()

    def check_draw(self):
        return not np.any(self._field == 0)

    def check_win(self, _id=None, return_how=False):
        t = (self._check_field == ((-self.next_action_id) if _id is None else _id)).float()
        k = self.kernel_len

        is_win, by, at = False, None, None

        by_row = self._row_kernel(t) >= k
        by_col = self._col_kernel(t) >= k
        by_diag = self._diag_kernel(t) >= k
        by_diag1 = self._diag1_kernel(t) >= k

        if torch.any(by_row):
            center = to_np(torch.nonzero(by_row)[0][2:])
            is_win, by, at = True, 'row', center + np.array([0, self.kernel_len // 2])
        if torch.any(by_col):
            center = to_np(by_col.nonzero()[0][2:])
            is_win, by, at = True, 'col', center + np.array([self.kernel_len // 2, 0])
        if torch.any(by_diag):
            center = to_np(by_diag.nonzero()[0][2:])
            is_win, by, at = True, 'diag', center + np.array([self.kernel_len // 2, self.kernel_len // 2])
        if torch.any(by_diag1):
            center = to_np(by_diag1.nonzero()[0][2:])
            is_win, by, at = True, 'diag1', center + np.array([self.kernel_len // 2, self.kernel_len // 2])

        return (is_win, by, at) if return_how else is_win

    def get_value(self, _id=None):
        depth = self.get_depth()
        value = None
        if depth > (self.kernel_len * 2 - 2):  # slightly speedups computation
            if self.check_win(_id=_id):
                value = WIN_VALUE
            elif depth == self._n ** 2:  # self.check_draw():
                value = DRAW_VALUE
        return value

    def get_state(self):
        return np.copy(self._field)

    def get_features(self):
        return np.copy(self._features)

    def get_running_features(self):
        return self._running_features

    def get_depth(self, field=None):
        field = self._field if field is None else field
        return np.count_nonzero(field)

    def get_size(self):
        return self._n

    def set_state(self, field):
        if not isinstance(field, np.ndarray):
            field = to_np(field)
        self._field = np.copy(np.float32(field))
        self._check_field = to_var(np.copy(self._field[None, None]), device=self.check_device)
        self.next_action_id = 1 if (self.get_depth() % 2 == 0) else -1
        self._features = self.field2features()
        self._running_features = to_var(np.copy(self._features), device=self.device)

    def show_field(self, field=None, colorbar=False):
        if field is None:
            field = self.get_state()
        if not isinstance(field, np.ndarray):
            field = to_np(field)

        plt.imshow(field, cmap='gray')
        if colorbar:
            plt.colorbar()
        plt.show()

    def draw_field(self, screen):
        for i in range(1, self._n):
            pygame.draw.line(screen, LINE_COLOR, (0, i * self.square_size), (WINDOW, i * self.square_size), 2)
            pygame.draw.line(screen, LINE_COLOR, (i * self.square_size, 0), (i * self.square_size, WINDOW), 2)

    def draw_figures(self, screen):
        circle_radius, circle_width = 200 // self._n, 40 // self._n
        cross_width, space = 50 // self._n, 10

        for row in range(self._n):
            for col in range(self._n):
                if self._field[row][col] == 1:
                    pygame.draw.line(
                        screen,
                        CROSS_COLOR,
                        (col * self.square_size + space, (row + 1) * self.square_size - space),
                        ((col + 1) * self.square_size - space, row * self.square_size + space),
                        cross_width
                    )

                    pygame.draw.line(
                        screen,
                        CROSS_COLOR,
                        (col * self.square_size + space, row * self.square_size + space),
                        ((col + 1) * self.square_size - space, (row + 1) * self.square_size - space),
                        cross_width
                    )
                elif self._field[row][col] == -1:
                    pygame.draw.circle(
                        screen,
                        CIRCLE_COLOR,
                        (int(col * self.square_size + self.square_size // 2),
                         int(row * self.square_size + self.square_size // 2)),
                        circle_radius,
                        circle_width
                    )

    def draw_winning_line(self, screen, by, at):
        row, col = at

        offset = self.square_size // 2

        posX = col * self.square_size + offset
        posY = row * self.square_size + offset

        if by == 'row':
            start = (posX - (self.kernel_len // 2) * self.square_size, posY)
            stop = (posX + (self.kernel_len // 2) * self.square_size, posY)
        elif by == 'col':
            start = (posX, posY - (self.kernel_len // 2) * self.square_size)
            stop = (posX, posY + (self.kernel_len // 2) * self.square_size)
        elif by == 'diag':
            start = (posX - (self.kernel_len // 2) * self.square_size, posY - (self.kernel_len // 2) * self.square_size)
            stop = (posX + (self.kernel_len // 2) * self.square_size, posY + (self.kernel_len // 2) * self.square_size)
        elif by == 'diag1':
            start = (posX + (self.kernel_len // 2) * self.square_size, posY - (self.kernel_len // 2) * self.square_size)
            stop = (posX - (self.kernel_len // 2) * self.square_size, posY + (self.kernel_len // 2) * self.square_size)
        else:
            raise IndexError('Unexpected winning position')

        pygame.draw.line(screen, WIN_LINE_COLOR, start, stop, 4)
