from tensorflow.keras import backend as K


def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """

    return (n + eps) / (d + eps)


def do_threshold(x, threshold=0.5):
    return K.cast(K.greater(x, threshold), dtype='float32')


def get_binary_img(x, threshold=0.5):
    x = K.sum(
        x[..., 1] + x[..., 2],  # foreground + borders
        axis=2
    )
    x = K.expand_dims(x, axis=-1)  # restore axis
    x = K.cast(K.greater(x, threshold), dtype='float32')
    return x


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
        y_pred = get_binary_img(y_pred)
        y_true = get_binary_img(y_true)

        inter = K.sum(K.squeeze(y_true * y_pred, axis=2), axis=1)
        union = K.sum(K.squeeze(y_true + y_pred, axis=2), axis=1) - inter
        
        return K.mean((inter + K.epsilon()) / (union + K.epsilon()))

    _f.__name__ = 'batch_metric-{}'.format('mean_IoU')
    return _f


def DSC(smooth=1.0, threshold=0.5):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    def _f(y_true, y_pred):
        y_pred = get_binary_img(y_pred)
        y_true = get_binary_img(y_true)

        inter = K.sum(K.squeeze(y_true * y_pred, axis=2), axis=1)
        union = K.sum(K.squeeze(y_true + y_pred, axis=2), axis=1)

        return K.mean((2.0 * inter + K.epsilon()) / (union + K.epsilon()))

    _f.__name__ = 'batch_metric-{}'.format('DSC')

    return _f
