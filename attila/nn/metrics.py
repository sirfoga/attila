from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Accuracy


def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """

    return (n + eps) / (d + eps)


def cast_threshold(x, threshold):
    return K.cast(K.greater(x, threshold), dtype='float32')


def get_binary_img(x, threshold=0.5):
    # todo hacky way to compute: consider only foreground + borders

    x = K.sum(  # sum all channels ...
        x[..., :-1],  # ... apart background (implicitely you also sum it, since it's a probability distribution)
        axis=-1
    )
    x = K.expand_dims(x, axis=-1)  # restore axis
    x = cast_threshold(x, threshold)
    return x  # (batch size, height, width, 1)


def get_intersection(y_true, y_pred):
    """ aka TP, assuming binary images that were removed the background """

    return K.sum(K.squeeze(y_true * y_pred, axis=-1), axis=-1)


def get_alls(y_true, y_pred):
    """ aka TP + TP + FP + FN, assuming binary images that were removed the background """

    return K.sum(K.squeeze(y_true + y_pred, axis=-1), axis=-1)


def is_from_batch(x):
    return len(x.shape) == 4  # bs size, x, y, ch


def mean_IoU(threshold=0.5):
    """
    - y_true is a 3D array. Each channel represents the ground truth BINARY channel
    - y_pred is a 3D array. Each channel represents the predicted BINARY channel
    """

    def _f(y_true, y_pred):
        if not is_from_batch(y_true):
            y_pred = K.expand_dims(y_pred, axis=0)  # add "batch axis"
        
        if not is_from_batch(y_pred):
            y_pred = K.expand_dims(y_pred, axis=0)  # add "batch axis"

        y_pred = get_binary_img(y_pred)
        y_true = get_binary_img(y_true)

        inter = get_intersection(y_true, y_pred)
        union = get_alls(y_true, y_pred) - inter

        if not is_from_batch(y_true):
            inter = inter.numpy().sum()
            union = union.numpy().sum()

        return eps_divide(inter, union)

    _f.__name__ = 'attila_metrics_{}'.format('mean_IoU')
    return _f


def DSC(smooth=1.0, threshold=0.5):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    def _f(y_true, y_pred):
        if not is_from_batch(y_true):
            y_pred = K.expand_dims(y_pred, axis=0)  # add "batch axis"
        
        if not is_from_batch(y_pred):
            y_pred = K.expand_dims(y_pred, axis=0)  # add "batch axis"

        y_pred = get_binary_img(y_pred)
        y_true = get_binary_img(y_true)

        inter = get_intersection(y_true, y_pred)
        alls = get_alls(y_true, y_pred)

        if not is_from_batch(y_true):
            inter = inter.numpy().sum()
            alls = alls.numpy().sum()

        return eps_divide(2.0 * inter + smooth, alls + smooth)

    _f.__name__ = 'attila_metrics_{}'.format('DSC')

    return _f


def calc_accuracy():
    m = Accuracy()

    def _f(y_true, y_pred):
        m.update_state(y_true, y_pred)
        return m.result().numpy()

    _f.__name__ = 'keras_metrics_{}'.format('accuracy')

    return _f
