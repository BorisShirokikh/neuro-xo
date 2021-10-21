import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dpipe.torch import to_np, to_device


DIRECTIONS = ('row', 'col', 'diag', 'diag1')


def get_check_kernel(kernel_len=5, direction='row'):
    if direction not in DIRECTIONS:
        raise ValueError(f'`direction` must be one of the {DIRECTIONS}, {direction} given')

    if 'diag' in direction:
        kernel_w = torch.Tensor(np.array([[np.eye(kernel_len)]]))
        kernel = nn.Conv2d(1, 1, kernel_len, bias=False)
        if '1' in direction:
            kernel.weight = nn.Parameter(kernel_w.transpose(-2, -1), requires_grad=False)
        else:
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        kernel.eval()
        return kernel
    else:
        kernel_w = torch.Tensor([[[[1] * kernel_len]]])
        if 'row' in direction:
            kernel = nn.Conv2d(1, 1, (1, kernel_len), bias=False)
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        else:
            kernel = nn.Conv2d(1, 1, (kernel_len, 1), bias=False)
            kernel.weight = nn.Parameter(kernel_w.transpose(-2, -1), requires_grad=False)
        kernel.eval()
        return kernel


class Field:
    def __init__(self, n=10, m=10, field=None, device='cpu'):
        self.device = device

        if field is not None:
            self._field = np.float32(field)
            self._n, self._m = self._field.shape
        else:
            self._n, self._m = n, m
            self._field = np.zeros((n, m), dtype='float32')

        # check win kernels:
        self._row_kernel = to_device(get_check_kernel(kernel_len=5, direction='row'), device=device)
        self._col_kernel = to_device(get_check_kernel(kernel_len=5, direction='col'), device=device)
        self._diag_kernel = to_device(get_check_kernel(kernel_len=5, direction='diag'), device=device)
        self._diag1_kernel = to_device(get_check_kernel(kernel_len=5, direction='diag1'), device=device)

    def make_move(self, i, j, _id):
        if self._field[i, j] != 0:
            raise IndexError('Accessing already marked cell.')

        self._field[i, j] = _id

    def backward_move(self, i, j):
        if self._field[i, j] == 0:
            raise IndexError('Clearing already empty cell.')

        self._field[i, j] = 0

    def check_draw(self):
        return not np.any(self._field == 0)

    def check_win(self, _id):
        t = to_device(torch.Tensor(self._field == _id)[None, None], device=self.device)

        if torch.any(self._row_kernel(t) >= 5):
            return True
        if torch.any(self._col_kernel(t) >= 5):
            return True
        if torch.any(self._diag_kernel(t) >= 5):
            return True
        if torch.any(self._diag1_kernel(t) >= 5):
            return True
        return False

    def get_state(self):
        return np.float32(self._field)

    def get_depth(self):
        return np.count_nonzero(self._field)

    def get_shape(self):
        return self._n, self._m

    def set_state(self, field):
        if not isinstance(field, np.ndarray):
            field = to_np(field)
        self._field = field

    def show_field(self, field=None, colorbar=False):
        if field is None:
            field = self.get_state()
        if not isinstance(field, np.ndarray):
            field = to_np(field)

        plt.imshow(field, cmap='gray')
        if colorbar:
            plt.colorbar()
        plt.show()
