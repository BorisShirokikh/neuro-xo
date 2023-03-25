import numpy as np
from numba import njit


@njit()
def rand_argmax(a):
    argmaxes = np.where(a == a.max())[0]
    return np.random.choice(argmaxes, replace=False) if (len(argmaxes) > 1) else argmaxes[0]


@njit()
def choose_idx(weights):
    wc = np.cumsum(weights)
    return np.searchsorted(wc, wc[-1] * np.random.rand(), side='right')
