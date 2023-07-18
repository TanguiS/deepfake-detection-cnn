def gpu_config():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    conf = tf.config.list_physical_devices('GPU')
    print(conf)


def cpu_config():
    from tensorflow.python.client import device_lib
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['ROCM_VISIBLE_DEVICES'] = '-1'
    print(device_lib.list_local_devices())
