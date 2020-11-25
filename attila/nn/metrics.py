import numpy as np
from tensorflow.keras import backend as K


# todo check with 3 channels
def eps_divide(n, d, eps=K.epsilon()):
    """ perform division using eps """
    return (n + eps) / (d + eps)


# todo check with 3 channels
def iou(y_true, y_pred, threshold=0.5):
    y_pred = K.cast(K.greater(y_pred, threshold), dtype='float32')
    inter = K.sum(y_true * y_pred)
    union = K.sum(y_true + y_pred) - inter
    return eps_divide(inter, union)


# todo check with 3 channels
def mean_IoU(y_true, y_pred, threshold=0.5):
    res = iou(y_true, y_pred, threshold=threshold)
    return K.mean(res)


# todo check with 3 channels
# todo refactor K.constant ...
def DSC(y_true, y_pred, smooth=1.0, threshold=0.5, axis=[1, 2, 3]):
    def _sum(x):
      return K.sum(x, axis=axis)
  
    y_pred = K.cast(K.greater(y_pred, threshold), dtype='float32')
    intersection = _sum(y_true * y_pred)
    union = _sum(y_true) + _sum(y_pred)
    return K.mean(eps_divide(2.0 * intersection, union + smooth, eps=smooth), axis=0)
