from abc import ABC
from pathlib import Path
from typing import Optional, Tuple, Union

import tensorflow
from tensorflow.keras import optimizers
from tensorflow.keras.applications import xception
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential

from models.base_model import ModelBase


class Xception(ModelBase, ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int], nb_epoch: int,
                 batch_size: int = 32, model_name: Optional[str] = None) -> None:
        super().__init__(models_dir, model_arch, input_shape, nb_epoch, batch_size, model_name)

        self.__setup()

    def __setup(self):
        if self.keras_model_path and self.keras_model_path.exists():
            self.log.info("Loading model...")
            self.__model = load_model(str(self.keras_model_path), compile=False)
            self.log.info(" > Done")
            return
        self.log.info("Creating model...")
        kwargs = {
            'include_top': True,
            'weights': None,
            'input_shape': self.input_shape,
            'classes': 2
        }
        self.__model: Model = xception.Xception(**kwargs)
        self.log.info(" > Done")

    def show_summary(self) -> None:
        self.__model.summary(print_fn=self.log.info)
        super().show_summary()

    def compile(self) -> None:
        loss = tensorflow.keras.losses.mean_squared_error
        if self.optimizer is None:
            self.log.info("Using new optimizer.")
            self.__model.compile(loss=loss,
                                 optimizer=optimizers.Adam(lr=1e-5),
                                 metrics=['accuracy'])
            return
        self.log.info(f"Loading optimizer and previous state from {self.optimizer}")
        self.__model.compile(loss=loss,
                             optimizer=optimizers.Adam.from_config(self.optimizer),
                             metrics=['accuracy'])

    def keras_model(self) -> Optional[Union[Model, Sequential]]:
        return self.__model


model = Xception
