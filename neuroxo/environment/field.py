from typing import Union

import torch
import torch.nn as nn
import numpy as np

# TODO: move from dpipe to own lib
from dpipe.torch import to_np, to_device, to_var


DIRECTIONS = ('row', 'col', 'diag', 'diag1')

EMPTY_ID = 0
X_ID = 1
O_ID = -1

WIN_VALUE = 1
DRAW_VALUE = 0


class Field:
    def __init__(self, n: int = 10, k: int = 5, field: Union[np.ndarray, torch.Tensor] = None,
                 device: str = 'cpu'):
        # Player0 = 'x', it's player_id (action_id) = 1 = X_ID
        # Player1 = 'o', it's player_id (action_id) = -1 = O_ID
        # For the empty field (default init) Player0 is the first to move:
        self._action_id = X_ID

        # Field init
        if field is None:  # Default (empty) field init
            self._field = np.zeros((n, n), dtype='float32')
            self._n = n
        else:  # Init field from passed array (np or torch)
            if not isinstance(field, np.ndarray):
                field = to_np(field)
            self._field = np.copy(np.float32(field))
            self._n = self._field.shape[0]
            self._action_id = self._get_action_id()

        self._k = k
        self._device = device
        self._features = self._field2features()
        self._running_features = self._get_running_features()
        self._check_kernels = {d: to_device(_get_check_kernel(d, k=k), device=device)
                               for d in DIRECTIONS}
        self._game_history = []

    # ################################################## SET SECTION #################################################

    def set_field(self, field, game_history=None):
        if not isinstance(field, np.ndarray):
            field = to_np(field)

        self._field = np.copy(np.float32(field))
        self._action_id = self._get_action_id()
        self._features = self._field2features()
        self._running_features = self._get_running_features()
        self._game_history = [] if game_history is None else game_history

    # ################################################# MOVE SECTION #################################################

    def make_move(self, i, j):
        if self._field[i, j] != EMPTY_ID:
            print(self._field, flush=True)
            print(i, j, flush=True)
            raise IndexError('Accessing already marked cell.')

        action_id = self._action_id

        self._field[i, j] = action_id
        self._running_features[0, 0, i, j] = 1
        self._features[0, 0, i, j] = 1
        self._update_action_id()
        self._game_history.append([i, j])

    def retract_move(self):
        if not self._game_history:
            raise ValueError('Trying to retract move with empty history.')

        i, j = self._game_history[-1]
        self._game_history = self._game_history[:-1]

        self._field[i, j] = EMPTY_ID
        self._running_features[0, 1, i, j] = 0
        self._features[0, 1, i, j] = 0
        self._update_action_id()

    # ######################################### WIN / DRAW / VALUE SECTION ###########################################

    def check_draw(self, is_win_checked=True):
        is_draw = self.get_depth() == self._n ** 2
        if is_draw and (not is_win_checked):
            is_draw = not self.check_win()
        return is_draw

    def check_win(self, return_how=False):
        """Returns bool `is_win` or tuple `(is_win, how, where)` if `return_how` is true."""
        t = self._running_features[:, [1], ...].float()

        # Sequential checks-returns and inverse (not return_how) typing to reduce computations:
        for d in DIRECTIONS:
            by_d = self._check_kernels[d](t) >= self._k
            if torch.any(by_d):
                return True if (not return_how) else (True, d, _get_win_center(by_d, d, self._k))

        return (False, None, None) if return_how else False

    # ################################################# GET SECTION ##################################################

    def get_depth(self, field=None):
        field = self._field if field is None else field
        return np.count_nonzero(field)

    def get_value(self):
        if self.get_depth() > (self._k * 2 - 2):  # slightly reduces computations
            if self.check_win():
                return WIN_VALUE
            elif self.check_draw():
                return DRAW_VALUE
            else:
                return None

    def get_field(self):
        return np.copy(self._field)

    def get_features(self):
        return np.copy(self._features)

    def get_running_features(self):
        return self._running_features

    def get_n(self):
        return self._n

    def get_k(self):
        return self._k

    def get_action_id(self):
        return self._action_id

    def get_opponent_action_id(self, action_id=None):
        action_id = self._action_id if action_id is None else action_id
        return X_ID if action_id == O_ID else O_ID

    def get_history(self):
        return self._game_history[:]

    def get_n_features(self):
        return self._features.shape[1]

    # ############################################### UTILS SECTION ##################################################

    def _get_action_id(self, field=None):
        field = self._field if field is None else field
        return X_ID if (self.get_depth(field) % 2 == 0) else O_ID

    def _update_action_id(self):
        self._action_id = self.get_opponent_action_id()
        self._running_features = torch.flip(self._running_features, dims=(1, ))
        self._features = np.flip(self._features, axis=1)

    def _field2features(self, field=None):
        field = self._field if field is None else field
        action_id = self._get_action_id(field)
        opponent_action_id = self.get_opponent_action_id(action_id=action_id)

        features = np.float32(np.concatenate((
            (field == action_id)[None, None],               # is player?      {0, 1}
            (field == opponent_action_id)[None, None],      # is opponent?    {0, 1}
        ), axis=1))
        return features

    def _get_running_features(self):
        return to_var(np.copy(self._field2features(field=self._field)), device=self._device)


def _get_check_kernel(direction, k=5):
    if direction not in DIRECTIONS:
        raise ValueError(f'`direction` must be one of the {DIRECTIONS}, {direction} given')

    if 'diag' in direction:  # 'diag' and 'diag1'
        kernel_size = k
        kernel_w = torch.tensor(np.array([[np.eye(k, dtype=np.float32)]]))
        kernel_w = kernel_w.flip(dims=(-1,)) if '1' in direction else kernel_w
    else:  # 'row' and 'col'
        kernel_size = (1, k) if direction == 'row' else (k, 1)
        kernel_w = torch.tensor([[[[1.] * k]]])
        kernel_w = kernel_w.transpose(-2, -1) if direction == 'col' else kernel_w

    kernel = nn.Conv2d(1, 1, kernel_size, bias=False)
    kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
    kernel.eval()
    return kernel


def _get_win_center(conv_map, by, k):
    by2shift = {'row': np.array([0, k // 2]), 'col': np.array([k // 2, 0]),
                'diag': np.array([k // 2, k // 2]), 'diag1': np.array([k // 2, k // 2])}
    return to_np(torch.nonzero(conv_map)[0][2:]) + by2shift[by]
