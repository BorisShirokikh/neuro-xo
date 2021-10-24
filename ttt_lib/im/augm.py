import numpy as np


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
