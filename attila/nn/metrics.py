import numpy as np
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

        scores.append(iou(true, pred))

    return scores



def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """
    
    return (n + eps) / (d + eps)


def get_intersection(y_true, y_pred):
    return np.sum(y_true * y_pred)  # TP


def get_union(y_true, y_pred):
    inter = get_intersection(y_true, y_pred)
    alls = np.sum(y_true + y_pred)  # TP + FP + TP + FN
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

    scores = metric_per_channel(y_true, y_pred, mean_IoU)
    return np.mean(scores)


def DSC(y_true, y_pred, smooth=1.0):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    tp = get_intersection(y_true, y_pred)
    tp_fp_fn = get_union(y_true, y_pred) + tp
    return eps_divide(2.0 * tp, tp + tp_fp_fn, eps=smooth)


def mean_DSC(y_true, y_pred, smooth=1.0):
    scores = metric_per_channel(y_true, y_pred, DSC)
    return np.mean(scores)


def batch_metric(metric):
    def _f(y_true, y_pred):
        true = y_true
        pred = y_pred

        batch_size = true.shape[0]
        scores = [
            metric(
                true[batch, ...],
                pred[batch, ...]
            )
            for batch in range(batch_size)
        ]
        return np.mean(scores)

    return _f