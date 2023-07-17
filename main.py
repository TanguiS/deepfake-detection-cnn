import argparser
from main_launcher.plot_history_launcher import decode_plot_args, launch_plot
from main_launcher.train_launcher import decode_trainer_args, launch_train


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


if __name__ == '__main__':

    args = argparser.args_parser()
    print("args : ", args)

    gpu_config()

    actions = {
        "train": (decode_trainer_args, launch_train),
        "plot": (decode_plot_args, launch_plot)
    }

    action = args["action"]

    try:
        decoder, launcher = actions[action]
    except KeyError:
        raise NotImplementedError(f"Action : '{action}' is not handled.")

    kwargs = decoder(args)
    launcher(**kwargs)
