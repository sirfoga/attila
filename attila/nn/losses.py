from keras import backend as K
import tensorflow as tf


def weighted_categorical_crossentropy(weights):
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        loss = y_true * tf.math.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss
