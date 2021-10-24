import os
from collections import defaultdict

import numpy as np


def get_running_mean_hash_table():
    return defaultdict(lambda: defaultdict(float))


def update_running_mean_hash_table(ht, h, v, key_m='running_mean', key_n='n'):
    ht[h][key_m] = ht[h][key_m] * ht[h][key_n] / (ht[h][key_n] + 1) + v / (ht[h][key_n] + 1)
    ht[h][key_n] += 1


def make_state_hashable(state):
    return tuple([tuple(arr) for arr in state.astype(int).tolist()])


def field2hashable(field):
    return make_state_hashable(field.get_state())


def sort_and_clear_running_mean_hash_table(ht, key_n='n', hashes_to_leave=1000):
    ht_new = get_running_mean_hash_table()
    hashes_sorted = sorted(ht, key=lambda _h: ht[_h][key_n], reverse=True)
    for h in hashes_sorted[:hashes_to_leave]:
        ht_new[h] = ht[h]
    return ht_new


def choose_model(path):
    models = [m for m in os.listdir(path) if 'model' in m]
    if len(models) == 0:
        return None
    elif 'model.pth' in models:
        return 'model.pth'
    else:
        n = np.max([int(m.strip('.pth').split('_')[-1]) for m in models])
        return f'model_{n}.pth'
