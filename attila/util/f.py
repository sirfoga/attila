import numpy as np


def compose_f(funcs):
    def _f(x):
        for f in funcs:
            x = f(x)
        return x

    return _f


def apply_f(lst, f, to_numpy):
    lst = [
        f(x)
        for x in lst
    ]

    if to_numpy:
        lst = np.array(lst)

    return lst


def apply_fs(lst, funcs, to_numpy=False):
    t = compose_f(funcs)
    return apply_f(lst, t, to_numpy=to_numpy)
