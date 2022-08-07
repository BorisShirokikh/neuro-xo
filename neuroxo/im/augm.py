import numpy as np
import torch


def get_dihedral_augm_torch_fn(return_inverse: bool = False):
    k = np.random.randint(low=0, high=4)
    flip = np.random.sample() > 0.5

    if flip:
        if return_inverse:
            return (lambda x: torch.flip(torch.rot90(x, k=k, dims=(-2, -1)), dims=(-1,)),
                    lambda x: torch.rot90(torch.flip(x, dims=(-1,)), k=-k, dims=(-2, -1)))
        else:
            return lambda x: torch.flip(torch.rot90(x, k=k, dims=(-2, -1)), dims=(-1, ))
    else:
        if return_inverse:
            return lambda x: torch.rot90(x, k=k, dims=(-2, -1)), lambda x: torch.rot90(x, k=-k, dims=(-2, -1))
        else:
            return lambda x: torch.rot90(x, k=k, dims=(-2, -1))


def get_dihedral_augm_numpy_fn(return_inverse: bool = False):
    k = np.random.randint(low=0, high=4)
    flip = np.random.sample() > 0.5

    if flip:
        if return_inverse:
            return (lambda x: np.flip(np.rot90(x, k=k, axes=(-2, -1)), axis=-1),
                    lambda x: np.rot90(np.flip(x, axis=-1), k=-k, axes=(-2, -1)))
        else:
            return lambda x: np.flip(np.rot90(x, k=k, axes=(-2, -1)), axis=-1)
    else:
        if return_inverse:
            return lambda x: np.rot90(x, k=k, axes=(-2, -1)), lambda x: np.rot90(x, k=-k, axes=(-2, -1))
        else:
            return lambda x: np.rot90(x, k=k, axes=(-2, -1))
