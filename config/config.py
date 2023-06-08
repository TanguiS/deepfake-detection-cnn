import tensorflow as tf


def config():
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True


def device() -> None:
    print(f'Is the GPU available? : {tf.test.is_gpu_available()}')
    config = tf.compat.v1.ConfigProto()
    with tf.compat.v1.Session(config=config) as sess:
        devices = sess.list_devices()
        print(devices)
