import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from dpipe.torch import to_np, to_device, to_var


DIRECTIONS = ('row', 'col', 'diag', 'diag1')


def get_check_kernel(kernel_len=5, direction='row'):
    if direction not in DIRECTIONS:
        raise ValueError(f'`direction` must be one of the {DIRECTIONS}, {direction} given')

    if 'diag' in direction:
        kernel_w = torch.tensor(np.array([[np.eye(kernel_len)]]))
        kernel = nn.Conv2d(1, 1, kernel_len, bias=False)
        if '1' in direction:
            kernel.weight = nn.Parameter(kernel_w.flip(dims=(-1, )), requires_grad=False)
        else:
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        kernel.eval()
        return kernel
    else:
        kernel_w = torch.tensor([[[[1] * kernel_len]]])
        if 'row' in direction:
            kernel = nn.Conv2d(1, 1, (1, kernel_len), bias=False)
            kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
        else:
            kernel = nn.Conv2d(1, 1, (kernel_len, 1), bias=False)
            kernel.weight = nn.Parameter(kernel_w.transpose(-2, -1), requires_grad=False)
        kernel.eval()
        return kernel


class Field:
    def __init__(self, n=10, m=10, field=None, device='cpu', check_device='cpu'):
        if field is not None:
            if not isinstance(field, np.ndarray):
                field = to_np(field)
            self._field = np.copy(np.float32(field))
            self._n, self._m = self._field.shape
        else:
            self._n, self._m = n, m
            self._field = np.zeros((n, m), dtype='float32')

        self.next_action_id = 1

        self.device = device
        self._features = self.field2features()
        self._running_features = to_var(np.copy(self._features), device=self.device)
        self.n_features = self._features.shape[1]

        self.check_device = check_device
        self._check_field = to_var(self._field[None, None], device=check_device)
        self._row_kernel = to_device(get_check_kernel(kernel_len=5, direction='row'), device=check_device)
        self._col_kernel = to_device(get_check_kernel(kernel_len=5, direction='col'), device=check_device)
        self._diag_kernel = to_device(get_check_kernel(kernel_len=5, direction='diag'), device=check_device)
        self._diag1_kernel = to_device(get_check_kernel(kernel_len=5, direction='diag1'), device=check_device)

    def _update_action_id(self):
        self.next_action_id = -self.next_action_id
        self._features = self._features[:, [1, 0, *range(2, self.n_features)], ...]
        self._running_features = self._running_features[:, [1, 0, *range(2, self.n_features)], ...]

    def field2features(self, field=None):
        field = self._field if field is None else np.copy(field)
        next_action_id = 1 if (self.get_depth(field) % 2 == 0) else -1
        features = np.float32(np.concatenate((
            (field == next_action_id)[None, None],        # is player?      {0, 1}
            (field == -next_action_id)[None, None],       # is opponent?    {0, 1}
            np.zeros_like(field)[None, None],             # zeros plane     {0}
            np.ones_like(field)[None, None],              # ones plane      {1}
            (field == 0)[None, None],                     # is legal move?  {0, 1}
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

    def check_win(self, _id=None):
        t = (self._check_field == ((-self.next_action_id) if _id is None else _id)).float()

        if torch.any(self._row_kernel(t) >= 5):
            return True
        if torch.any(self._col_kernel(t) >= 5):
            return True
        if torch.any(self._diag_kernel(t) >= 5):
            return True
        if torch.any(self._diag1_kernel(t) >= 5):
            return True
        return False

    def get_value(self, _id=None):
        value = None
        if self.get_depth() > 8:  # slightly speedups computation
            if self.check_win(_id=_id):
                value = 1
            elif self.check_draw():
                value = 0.5
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

    def get_shape(self):
        return self._n, self._m

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
