import pickle
from abc import ABC
from pathlib import Path
from typing import Optional, Tuple, Union

import keras
from keras import optimizers, layers
from keras.applications import vgg19
from keras.engine.saving import load_model

from models.base_model import ModelBase


class VGG19(ModelBase, ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int], nb_epoch: int,
                 batch_size: int = 32, model_name: Optional[str] = None) -> None:
        super().__init__(models_dir, model_arch, input_shape, nb_epoch, batch_size, model_name)

        self.__setup()

    def __setup(self):
        if self.model_hdf5_file_path and self.model_hdf5_file_path.exists():
            self.__model = load_model(str(self.model_hdf5_file_path), compile=False)
            return
        kwargs = {
            'include_top': True,
            'weights': None,
            'input_shape': self.shape,
            'classes': 2
        }
        self.__model: keras.Model = vgg19.VGG19(**kwargs)

    def show_summary(self) -> None:
        self.__model.summary(print_fn=self.log.info)

    def compile(self) -> None:
        if self.optimizer is None:
            self.__model.compile(loss='binary_crossentropy',
                                 optimizer=optimizers.Adam(),
                                 metrics=['accuracy'])
            return
        self.__model.compile(loss='binary_crossentropy',
                             optimizer=optimizers.Adam.from_config(self.optimizer),
                             metrics=['accuracy'])

    def model(self) -> Optional[Union[keras.Model, keras.Sequential]]:
        return self.__model


Model = VGG19
