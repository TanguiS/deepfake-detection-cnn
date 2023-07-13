from pathlib import Path
from typing import Dict, Tuple, Optional

import argparser
from bench.plot_history import plot_training_results
import tensorflow as tf


def decode_trainer_args(args: Dict[str, any]):
    input_shape = decode_shape(args['shape'], args['grayscale'])
    kwargs = {
        'input_shape': input_shape,
        'root_face_folder': args['root_face_folder'],
        'df_faces_path': args['df_faces_path'],
        'output_folder': args['output_folder'],
        'arch': args['arch'],
        'batch_size': args['batch_size'],
        'target_epoch': args['target_epoch'],
        'model_name': args['model_name'],
        'seed': args['seed'],
        'distribution': args['distribution']
    }
    return kwargs


def launch_train(
        root_face_folder: Path,
        df_faces_path: Path,
        arch: str,
        input_shape: Tuple[int, int, int],
        batch_size: int,
        target_epoch: int,
        output_folder: Path,
        distribution: str,
        seed: int,
        model_name: Optional[str]
) -> None:
    workspace = output_folder.joinpath(arch)
    workspace.mkdir(exist_ok=True)
    logger = Logger(workspace)

    data = DataLoader(root_face_folder, df_faces_path, input_shape[0], seed, arch, batch_size, distribution)
    data.summary()

    model: ModelBase = models.import_model(arch)(
        models_dir=output_folder,
        model_arch=arch,
        input_shape=input_shape,
        nb_epoch=1,
        batch_size=batch_size,
        model_name=model_name
    )
    model.show_summary()

    trainer = Trainer(model, data, batch_size)

    trainer.run(target_epoch)


def decode_plot_args(args: Dict[str, any]):
    kwargs = {
        'train_csv_path': Path(args['train_history_csv_path']),
        'val_csv_path': Path(args['validation_history_csv_path'])
    }
    return kwargs


def launch_plot(train_csv_path: Path, val_csv_path: Path) -> None:
    plot_training_results(train_csv_path, val_csv_path)


def decode_shape(shape: int, grayscale: bool):
    channels = 3
    if grayscale:
        channels = 1
        raise NotImplementedError(f"Grayscale is not implemented yet.")
    input_shape = (shape, shape, channels)

    return input_shape


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    conf = tf.config.list_physical_devices('GPU')

    print(conf)

    from log_io.logger import Logger
    from models import ModelBase
    from train.Trainer import Trainer
    from pathlib import Path
    import models
    from data.DataLoader import DataLoader

    args = argparser.args_parser()
    print("args : ", args)

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
