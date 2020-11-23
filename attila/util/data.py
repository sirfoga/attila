import numpy as np


def is_numpy_array(x):
    return isinstance(x, type(np.zeros(1)))


def is_lst(x):
    return isinstance(x, type([]))
