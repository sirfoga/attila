import numpy as np


def is_numpy_array(x):
    return isinstance(x, type(np.zeros(1)))


def is_lst(x):
    return isinstance(x, type([]))


def dict2numpy(x):
    return {
        k: np.float32(val)
        for k, val in x.items()
    }