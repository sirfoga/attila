import tensorflow as tf


def get_avail_gpu():
    return tf.config.list_physical_devices('GPU')


def how_many_avail_gpu():
    return len(get_avail_gpu())


def are_gpu_avail():
    return len(get_avail_gpu()) >= 1
