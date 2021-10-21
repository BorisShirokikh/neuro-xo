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


def flip_augm(x, p=0.5):
    if np.random.rand() < p:
        flip_axis = np.random.choice((0, 1))
        x = np.flip(x, axis=flip_axis)
    return x


def rot_augm(x, p=0.75):
    if np.random.rand() < p:
        x = np.rot90(x, k=np.random.randint(1, 4), axes=(0, 1))
    return x


def augm_spatial(x):
    return rot_augm(flip_augm(x))
