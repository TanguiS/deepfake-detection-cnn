from abc import ABC
from pathlib import Path
from typing import Optional, Tuple

import tensorflow
from tensorflow.keras.applications import resnet_v2

from models.ModelBase import ModelBase


class ResNet152V2(ModelBase, ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int], nb_epoch: int,
                 batch_size: int = 32, model_name: Optional[str] = None) -> None:
        super().__init__(models_dir, model_arch, input_shape, nb_epoch, batch_size, model_name)
        self.__setup_model__()

    def __get_architecture_class__(self) -> any:
        return resnet_v2.ResNet152V2

    def __get_losses_metric__(self):
        return tensorflow.keras.losses.categorical_crossentropy


model = ResNet152V2
