from tensorflow.keras import backend as K


def get_foreground(x):
    fore = x[..., 1]
    borders = x[..., 2]

    return fore + borders


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


def mean_IoU(threshold=0.5):
    """
    - y_true is a 3D array. Each channel represents the ground truth BINARY channel
    - y_pred is a 3D array. Each channel represents the predicted BINARY channel
    """

    # todo mean over ALL classes
    # y_true = do_threshold(get_foreground(y_true))
    # y_pred = do_threshold(get_foreground(y_pred))
    # return iou(y_true, y_pred)

    def _f(y_true, y_pred):
        y_pred = K.cast(K.greater(y_pred[..., 1] + y_pred[..., 2], threshold), dtype='float32')  # do not count background
        y_true = K.cast(K.greater(y_true[..., 1] + y_true[..., 2], threshold), dtype='float32')

        print(y_pred.shape)

        inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
        union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1) - inter
        
        return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

    _f.__name__ = 'batch_metric-{}'.format('mean_IoU')
    return _f


def DSC(smooth=1.0, threshold=0.5):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    def _f(y_true, y_pred):
        y_pred = K.cast(K.greater(y_pred[..., 1] + y_pred[..., 2], threshold), dtype='float32')  # do not count background
        y_true = K.cast(K.greater(y_true[..., 1] + y_true[..., 2], threshold), dtype='float32')

        inter = K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)
        union = K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1)

        return K.mean((2.0 * inter + K.epsilon()) / (union + K.epsilon()))

    _f.__name__ = 'batch_metric-{}'.format('DSC')

    return _f
