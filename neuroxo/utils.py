from pathlib import Path

import numpy as np
import torch


def get_repo_root():
    import neuroxo
    return Path(neuroxo.__file__).parent.parent


def flush(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)


def update_lr(optimizer: torch.optim.Optimizer, epoch2lr: dict, epoch: int):
    try:
        optimizer.param_groups[0]['lr'] = epoch2lr[epoch]
    except (KeyError, TypeError):  # (no such epochs in the dict, dict is None)
        pass


def s2hhmmss(s: float):
    s = int(s)
    hh = s // 3600
    mm = (s % 3600) // 60
    ss = s % 60
    return f'{hh:02d}:{mm:02d}:{ss:02d}'


def n2p_binomial_test(n: int):
    return 1.96 * np.sqrt(0.25 / n) + 0.5
