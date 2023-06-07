import tensorflow as tf


def config():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
