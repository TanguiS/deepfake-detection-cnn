from pathlib import Path
from typing import Dict, Tuple, Optional

import models
from data.DataLoader import DataLoader
from log_io.logger import Logger

from util import decode_shape
from models import ModelBase
from train.Trainer import Trainer


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
    from config import config
    config.gpu_config()
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
