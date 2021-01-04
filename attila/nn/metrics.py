from keras import backend as K
from tensorflow.keras.metrics import Accuracy
from tensorflow.keras.layers import Cropping2D

from attila.patch.layers import CenterCrop



def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """

    return (n + eps) / (d + eps)


def cast_threshold(x, threshold):
    return K.cast(K.greater(x, threshold), dtype='float32')


def get_binary_img(x, threshold=0.5, center_crop=0, n_classes=2):
    if center_crop > 0:
        x = CenterCrop(center_crop, center_crop)(x)

    x = K.sum(  
        x[..., :n_classes],  # consider only first N classes
        axis=-1
    )
    x = K.expand_dims(x, axis=-1)  # restore axis
    x = cast_threshold(x, threshold)
    return x  # (batch size, height, width, 1)


def get_intersection(y_true, y_pred):
    """ aka TP, assuming binary images that were removed the background """

    return K.sum(K.sum(K.squeeze(y_true * y_pred, axis=3), axis=2), axis=1)

def get_alls(y_true, y_pred):
    """ aka TP + TP + FP + FN, assuming binary images that were removed the background """

    return K.sum(K.sum(K.squeeze(y_true + y_pred, axis=3), axis=2), axis=1)


def is_from_batch(x):
    return len(x.shape) == 4  # bs size, x, y, ch


def has_missing_batch(x):
    return not is_from_batch(x) and x.shape[-1] < 10  # last shape is channel


def has_missing_channel(x):
    return not is_from_batch(x) and x.shape[-1] > 10


def fix_input(x):
    if has_missing_batch(x):
        x = K.expand_dims(x, axis=0)  # add "batch axis"
    
    if has_missing_channel(x):
        x = K.expand_dims(x, axis=-1)  # add "channel axis"

    return x

def mean_IoU(threshold=0.5, center_crop=0):
    """
    - y_true is a 3D array. Each channel represents the ground truth BINARY channel
    - y_pred is a 3D array. Each channel represents the predicted BINARY channel
    """

    def _f(y_true, y_pred):
        y_true = fix_input(y_true)
        y_pred = fix_input(y_pred)

        y_true = get_binary_img(
            y_true,
            threshold=threshold,
            center_crop=center_crop
        )
        y_pred = get_binary_img(
            y_pred,
            threshold=threshold,
            center_crop=center_crop
        )

        inter = get_intersection(y_true, y_pred)
        union = get_alls(y_true, y_pred) - inter

        batch_metric = eps_divide(inter, union)
        return K.mean(batch_metric)

    _f.__name__ = 'attila_metrics_{}'.format('mean_IoU')
    return _f


def DSC(smooth=1.0, threshold=0.5):
    """
    - y_true is a 2D array representing the ground truth BINARY image
    - y_pred is a 2D array representing the predicted BINARY image
    """

    def _f(y_true, y_pred):
        y_true = fix_input(y_true)
        y_pred = fix_input(y_pred)

        y_true = get_binary_img(
            y_true,
            threshold=threshold,
            center_crop=center_crop
        )
        y_pred = get_binary_img(
            y_pred,
            threshold=threshold,
            center_crop=center_crop
        )

        inter = get_intersection(y_true, y_pred)
        alls = get_alls(y_true, y_pred)

        if not is_from_batch(y_true):
            inter = inter.numpy().sum()
            alls = alls.numpy().sum()

        batch_metric = eps_divide(2.0 * inter + smooth, alls + smooth)
        return K.mean(batch_metric)

    _f.__name__ = 'attila_metrics_{}'.format('DSC')

    return _f


def calc_accuracy():
    m = Accuracy()

    def _f(y_true, y_pred):
        m.update_state(y_true, y_pred)
        return m.result().numpy()

    _f.__name__ = 'keras_metrics_{}'.format('accuracy')

    return _f
