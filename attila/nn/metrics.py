from tensorflow.keras import backend as K


def metric_per_channel(y_true, y_pred, metric):
    """
    - y_true is a 3D array. Each channel represents the ground truth BINARY channel
    - y_pred is a 3D array. Each channel represents the predicted BINARY channel
    - metric is the metric function to apply for each channel
    """

    n_channels = y_true.shape[-1]
    scores = []

    for channel in range(n_channels):
        true = y_true[..., channel]
        pred = y_pred[..., channel]

        scores.append(metric(true, pred))

    return scores


def get_foreground(x):
    fore = x[..., 1]
    borders = x[..., 2]

    return K.sum(fore, borders)


def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """

    return (n + eps) / (d + eps)


def do_threshold(x, threshold=0.5):
    return K.cast(K.greater(x, threshold), dtype='float32')


def get_intersection(y_true, y_pred):
    elem_wise_prod = y_true * y_pred
    return K.sum(elem_wise_prod)  # TP


def get_union(y_true, y_pred):
    inter = get_intersection(y_true, y_pred)
    alls = K.sum(y_true + y_pred)  # TP + FP + TP + FN
    return alls - inter  # TP + FP + FN


def iou(y_true, y_pred):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    inter = get_intersection(y_true, y_pred)
    union = get_union(y_true, y_pred)
    return eps_divide(inter, union)


def mean_IoU(y_true, y_pred):
    """
    - y_true is a 3D array. Each channel represents the ground truth BINARY channel
    - y_pred is a 3D array. Each channel represents the predicted BINARY channel
    """

    # todo mean over ALL classes
    y_true = do_threshold(get_foreground(y_true))
    y_pred = do_threshold(get_foreground(y_pred))
    return iou(y_true, y_pred)


def DSC(y_true, y_pred, smooth=1.0, threshold=0.5):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    y_true = do_threshold(get_foreground(y_true))
    y_pred = do_threshold(get_foreground(y_pred))

    tp = get_intersection(y_true, y_pred)
    tp_fp_fn = get_union(y_true, y_pred) + tp

    return eps_divide(2.0 * tp, tp + tp_fp_fn, eps=smooth)


def batch_metric(metric):
    def _f(y_true, y_pred):
        batch_size = y_true.shape[0]
        if batch_size is None:
            batch_size = 1

        scores = [
            metric(
                y_true[batch, ...],
                y_pred[batch, ...]
            )
            for batch in range(batch_size)
        ]
        scores = K.cast(scores, dtype='float32')
        return K.mean(scores)

    _f.__name__ = 'batch_metric-{}'.format(metric.__name__)
    return _f
