def config():
    import tensorflow as tf

    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    conf = tf.config.list_physical_devices('GPU')
    print(conf)
