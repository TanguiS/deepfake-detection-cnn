

import plaidml.keras

from pathlib import Path
from typing import Dict, Tuple
import argparser
from bench.plot_history import plot_training_results


def decode_trainer_args(args: Dict[str, any]):
    input_shape = decode_shape(args['shape'], args['grayscale'])
    kwargs = {
        'input_shape': input_shape,
        'root_face_folder': args['root_face_folder'],
        'df_faces_path': args['df_faces_path'],
        'output_folder': args['output_folder'],
        'output_dataframe': args['output_dataframe'],
        'arch': args['arch'],
        'batch_size': args['batch_size'],
        'target_epoch': args['target_epoch']
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
        output_dataframe: Path
) -> None:
    workspace = output_folder.joinpath(arch)
    workspace.mkdir(exist_ok=True)
    logger = Logger(workspace)

    data = DataLoader(root_face_folder, df_faces_path, input_shape[0], output_dataframe, batch_size)
    data.summary()

    model: ModelBase = models.import_model(arch)(
        models_dir=output_folder,
        model_arch=arch,
        input_shape=input_shape,
        nb_epoch=1,
        batch_size=batch_size,
        model_name=None
    )
    model.show_summary()

    trainer = Trainer(model, data, batch_size)

    trainer.run(target_epoch)


def decode_plot_args(args: Dict[str, any]):
    kwargs = {
        'train_csv_path': args['train_history_csv_path'],
        'val_csv_path': args['validation_history_csv_path']
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
    plaidml.keras.install_backend()

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
