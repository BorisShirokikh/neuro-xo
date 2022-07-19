from typing import Union

import torch
import torch.nn as nn
import numpy as np

from dpipe.torch import to_np, to_device, to_var


DIRECTIONS = ('row', 'col', 'diag', 'diag1')

EMPTY_ID = 0
X_ID = 1
O_ID = -1

WIN_VALUE = 1
DRAW_VALUE = 0


class Field:
    def __init__(self, n: int = 10, kernel_len: int = 5, field: Union[np.ndarray, torch.Tensor] = None,
                 device: str = 'cpu', check_device: str = 'cpu'):
        # Player0 = 'x', it's player_id (action_id) = 1 = X_ID
        # Player1 = '0', it's player_id (action_id) = -1 = O_ID
        # For the empty field (default init) Player0 is the first to move:
        self._action_id = X_ID

        # Field init
        if field is None:  # Default (empty) field init
            self._field = np.zeros((n, n), dtype='float32')
            self._n = n
        else:  # Init field from passed array (np or torch)
            if not isinstance(field, np.ndarray):
                field = to_np(field)
            _check_numpy_field(field)
            self._field = np.copy(np.float32(field))
            self._n = self._field.shape[0]
            self._action_id = self._get_action_id()

        # Condition to win init
        self.kernel_len = kernel_len

        # Check win algorithm init
        self.check_device = check_device
        self._check_field = self._get_check_field()
        self._row_kernel = to_device(_get_check_kernel('row', kernel_len=kernel_len), device=check_device)
        self._col_kernel = to_device(_get_check_kernel('col', kernel_len=kernel_len), device=check_device)
        self._diag_kernel = to_device(_get_check_kernel('diag', kernel_len=kernel_len), device=check_device)
        self._diag1_kernel = to_device(_get_check_kernel('diag1', kernel_len=kernel_len), device=check_device)

        # Features
        self._features = self._field2features()
        self._n_features = self._features.shape[1]

        # Torch running values (to speed up computations):
        self.device = device
        self._running_features = self._get_running_features()

    # ################################################## SET SECTION #################################################

    def set_field(self, field):
        if not isinstance(field, np.ndarray):
            field = to_np(field)
        _check_numpy_field(field)

        self._field = np.copy(np.float32(field))
        self._check_field = self._get_check_field()
        self._action_id = self._get_action_id()
        self._features = self._field2features()
        self._running_features = self._get_running_features()

    # ################################################# MOVE SECTION #################################################

    def make_move(self, i, j, action_id=None):
        if self._field[i, j] != 0:
            raise IndexError('Accessing already marked cell.')

        action_id = self._action_id if action_id is None else action_id

        self._field[i, j] = action_id
        self._check_field[0, 0, i, j] = action_id

        self._features[0, 0, i, j] = 1
        self._features[0, -1, i, j] = 0

        self._running_features[0, 0, i, j] = 1
        self._running_features[0, -1, i, j] = 0

        self._update_action_id()

    def retract_move(self, i, j):
        if self._field[i, j] == 0:
            raise IndexError('Clearing already empty cell.')

        self._field[i, j] = EMPTY_ID
        self._check_field[0, 0, i, j] = EMPTY_ID

        self._features[0, 0, i, j] = 0
        self._features[0, -1, i, j] = 1

        self._running_features[0, 0, i, j] = 0
        self._running_features[0, -1, i, j] = 1

        self._update_action_id()

    # ######################################### WIN / DRAW / VALUE SECTION ###########################################

    def check_draw(self, is_win_checked=True):
        is_draw = self._get_depth() == self._n ** 2
        if is_draw and (not is_win_checked):
            is_draw = not self.check_win()
        return is_draw

    def check_win(self, action_id=None, return_how=False):
        """Returns bool `is_win` or tuple `(is_win, how, where)` if `return_how` is true."""
        action_id = self.get_opponent_action_id() if action_id is None else action_id
        t = (self._check_field == action_id).float()
        k = self.kernel_len

        # Sequential checks-returns and inverse (not return_how) typing to reduce computations:

        by_row = self._row_kernel(t) >= k
        if torch.any(by_row):
            return True if (not return_how) else (True, 'row', _get_win_center(by_row, 'row', k))

        by_col = self._col_kernel(t) >= k
        if torch.any(by_col):
            return True if (not return_how) else (True, 'col', _get_win_center(by_col, 'col', k))

        by_diag = self._diag_kernel(t) >= k
        if torch.any(by_diag):
            return True if (not return_how) else (True, 'diag', _get_win_center(by_diag, 'diag', k))

        by_diag1 = self._diag1_kernel(t) >= k
        if torch.any(by_diag1):
            return True if (not return_how) else (True, 'diag1', _get_win_center(by_diag1, 'diag1', k))

        return (False, None, None) if return_how else False

    # ################################################# GET SECTION ##################################################

    def get_value(self):
        # TODO: probably we need to access a custom `field` given by arg.
        if self._get_depth() > (self.kernel_len * 2 - 2):  # slightly reduces computations
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

    def get_action_id(self):
        return self._action_id

    def get_opponent_action_id(self, action_id=None):
        action_id = self._action_id if action_id is None else action_id
        return X_ID if action_id == O_ID else O_ID

    # ############################################### UTILS SECTION ##################################################

    def _get_depth(self, field=None):
        field = self._field if field is None else field
        return np.count_nonzero(field)

    def _get_action_id(self, field=None):
        field = self._field if field is None else field
        return X_ID if (self._get_depth(field) % 2 == 0) else O_ID

    def _update_action_id(self):
        self._action_id = self.get_opponent_action_id()
        # TODO: could it be done via np/torch functions?
        self._features = self._features[:, [1, 0, *range(2, self._n_features)], ...]
        self._running_features = self._running_features[:, [1, 0, *range(2, self._n_features)], ...]

    def _get_check_field(self):
        return to_var(self._field[None, None], device=self.check_device)

    def _get_running_features(self):
        return to_var(np.copy(self._features), device=self.device)

    def _field2features(self, field=None):
        field = self._field if field is None else field
        action_id = self._get_action_id(field)
        opponent_action_id = self.get_opponent_action_id(action_id=action_id)

        features = np.float32(np.concatenate((
            (field == action_id)[None, None],               # is player?      {0, 1}
            (field == opponent_action_id)[None, None],      # is opponent?    {0, 1}
            np.zeros_like(field)[None, None],               # zeros plane     {0}
            np.ones_like(field)[None, None],                # ones plane      {1}
            (field == 0)[None, None],                       # is legal move?  {0, 1}
        ), axis=1))
        return features


def _check_numpy_field(field: np.ndarray):
    if len(field.shape) != 2:
        raise ValueError(f'Field could be initialized only with 2-dim arrays; '
                         f'however, {len(field.shape)}-dim array is given')
    if field.shape[0] != field.shape[1]:
        raise ValueError(f'Working with square fields only; however, shape `{field.shape}` is given.')


def _get_check_kernel(direction, kernel_len=5):
    if direction not in DIRECTIONS:
        raise ValueError(f'`direction` must be one of the {DIRECTIONS}, {direction} given')

    if 'diag' in direction:  # 'diag' and 'diag1'
        kernel_size = kernel_len
        kernel_w = torch.tensor(np.array([[np.eye(kernel_len, dtype=np.float32)]]))
        kernel_w = kernel_w.flip(dims=(-1,)) if '1' in direction else kernel_w
    else:  # 'row' and 'col'
        kernel_size = (1, kernel_len) if direction == 'row' else (kernel_len, 1)
        kernel_w = torch.tensor([[[[1.] * kernel_len]]])
        kernel_w = kernel_w.transpose(-2, -1) if direction == 'col' else kernel_w

    kernel = nn.Conv2d(1, 1, kernel_size, bias=False)
    kernel.weight = nn.Parameter(kernel_w, requires_grad=False)
    kernel.eval()
    return kernel


def _get_win_center(conv_map, by, k):
    by2shift = {'row': np.array([0, k // 2]), 'col': np.array([k // 2, 0]),
                'diag': np.array([k // 2, k // 2]), 'diag1': np.array([k // 2, k // 2])}
    return to_np(torch.nonzero(conv_map)[0][2:]) + by2shift[by]
