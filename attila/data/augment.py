import numpy as np

from attila.util.f import apply_f


def flip():
    """ vertical flipping (prev: sun on top, after: sun on bottom) """

    def _f(x):
        """ np.array of shape (n pixels, m pixels, c channels), e.g (256, 256, 3)"""

        return np.flip(x, axis=0)

    return _f


def flop():
    """ horizontal flipping (prev: sun on left, after: sun on right) """

    def _f(x):
        """ np.array of shape (n pixels, m pixels, c channels), e.g (256, 256, 3)"""

        return np.flip(x, axis=1)

    return _f


def do_augmentation(data, augm):
    data_augm = apply_f(
        data,
        augm,
        to_numpy=True
    )
    return np.append(data, data_augm, axis=0)


def do_augmentations(X, y, augmentations):
    for augm in augmentations:
        X = do_augmentation(X, augm)
        y = do_augmentation(y, augm)

    return X, y
