from collections import defaultdict


def get_running_mean_hash_table():
    return defaultdict(lambda: defaultdict(float))


def update_running_mean_hash_table(ht, h, v, key_m='running_mean', key_n='n'):
    ht[h][key_m] = ht[h][key_m] * ht[h][key_n] / (ht[h][key_n] + 1) + v / (ht[h][key_n] + 1)
    ht[h][key_n] += 1


def field2state(field):
    return tuple([tuple(arr) for arr in field.get_state().astype(int).tolist()])


def sort_and_clear_running_mean_hash_table(ht, key_n='n', hashes_to_leave=1000):
    ht_new = get_running_mean_hash_table()
    hashes_sorted = sorted(ht, key=lambda _h: ht[_h][key_n], reverse=True)
    for h in hashes_sorted[:hashes_to_leave]:
        ht_new[h] = ht[h]
    return ht_new
