import numpy as np
from sklearn.preprocessing import minmax_scale
from skimage.segmentation import find_boundaries

from attila.util.f import apply_fs


def normalize_transformation(feature_range):
    def _f(x):
        shape = x.shape
        x = minmax_scale(x.ravel(), feature_range=feature_range)
        x = x.reshape(shape)  # original size
        return x

    return _f


def crop_center_transformation(shape):
    (height, width) = shape

    def get_start_point(dim, cropping):
        return dim // 2 - cropping // 2

    def get_end_point(start, cropping):
        return start + cropping

    def _f(img):
        n_dim = len(img.shape)

        if n_dim == 2:
            (y, x) = img.shape
        elif n_dim == 3:
            (y, x, _) = img.shape  # n channels not wanted
        else:  # assuming tensor (n_dim = 4)
            (_, y, x, _) = img.shape  # batch size AND n channels not wanted
        
        (start_x, start_y) = (get_start_point(x, width), get_start_point(y, height))
        (end_x, end_y) = (get_end_point(start_x, width), get_end_point(start_y, height))
        
        
        if n_dim == 2:
            return img[start_y: end_y, start_x: end_x]
        elif n_dim == 3:
            return img[start_y: end_y, start_x: end_x, :]
        else:  # assuming tensor (n_dim = 4)
            return img[:, start_y: end_y, start_x: end_x, :]

    return _f


def rm_percentiles_transformation(min_p=0.0, max_p=100.0):
    def _f(x):
        shape = x.shape
        x = x.ravel()
        new_min, new_max = np.percentile(x, [min_p, max_p])
        x[x < new_min] = new_min
        x[x > new_max] = new_max
        return x.reshape(shape)

    return _f


def add_dim():
    def _f(x):
        new_dim_index = len(x.shape)
        return np.expand_dims(x, new_dim_index)

    return _f


def get_borders(img):
    """ gets borders of img """

    return find_boundaries(img, mode='inner')


def get_background(foreground):
    """ gets background of grayscale img """

    out = foreground.copy()
    out[out > 1] = 1  # threshold
    out[out < 0] = 0  # threshold
    out = 1 - out  # flip (1 is background pixel now)

    return out


def get_foreground(img, borders):
    """ gets foreground of grayscale img """

    out = img.copy()

    out = out + borders
    out[out > 1] = 0  # borders
    out[out < 1] = 0  # background

    return out


def img2channels():
    """ splits grayscale 2d img -> channels: background, foreground, borders ..."""

    def _f(x):
        borders = get_borders(x)
        foreground = get_foreground(x, borders)
        background = get_background(foreground + borders)

        out = np.append(
            add_dim()(foreground),
            add_dim()(borders),
            axis=-1
        )
        out = np.append(
            out,
            add_dim()(background),
            axis=-1
        )

        return out  # each pixels is a one-hot vector

    return _f


def do_transformations(data, transformations):
    return apply_fs(data, transformations, to_numpy=True)
