from abc import ABC
from pathlib import Path
from typing import Optional, Tuple

import tensorflow
from tensorflow.keras.applications import xception

from models.ModelBase import ModelBase


class Xception(ModelBase, ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int],
                 model_name: Optional[str] = None) -> None:
        super().__init__(models_dir, model_arch, input_shape, model_name)
        self.__setup_model__()

    def __get_losses_metric__(self) -> any:
        return tensorflow.keras.losses.categorical_crossentropy

    def __get_architecture_class__(self) -> any:
        return xception.Xception


model = Xception
