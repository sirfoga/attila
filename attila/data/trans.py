import numpy as np
from sklearn.preprocessing import minmax_scale


def normalize_transformation(feature_range):
    def _f(x):
        shape = x.shape
        x = minmax_scale(x.ravel(), feature_range=feature_range)
        x = x.reshape(shape)    # original size
        return x

    return _f


def crop_center_transformation(shape):
    height, width, *_ = shape    # 3rd dim not needed

    def get_start_point(dim, cropping):
        return dim // 2 - cropping // 2

    def get_end_point(start, cropping):
        return start + cropping

    def _f(img):
        y, x, *_ = img.shape    # n channels not wanted
        (start_x, start_y) = (get_start_point(x, width), get_start_point(y, height))
        (end_x, end_y) = (get_end_point(start_x, width), get_end_point(start_y, height))
        return img[start_y: end_y, start_x: end_x, ...]

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
        x = np.expand_dims(x, new_dim_index)
        return x

    return _f


def compose_transformations(transformations):
    def _f(x):
        for t in transformations:
            x = t(x)
        return x

    return _f


def apply_transformations(lst, transformations):
    t = compose_transformations(transformations)
    return [
        t(x) for x in lst
    ]


def do_transformations(X, y, transformations):
    X = apply_transformations(X, transformations)
    y = apply_transformations(y, transformations)
    return X, y
