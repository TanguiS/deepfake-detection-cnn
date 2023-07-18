import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional, Union

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model, load_model, Sequential

from log_io import logger
from models.util import new_model_format, decode_model_name, find_models, choose_models, model_first_run_choose_name


# Model name format : {models arch}_{model-name}_{dim: xy_channel}.*

class ModelBase(ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int],
                 model_name: Optional[str] = None) -> None:
        super().__init__()

        self.__optimizer = None
        self.__model: Optional[Model] = None
        self.__input_shape = input_shape
        self.__models_dir = models_dir
        self.__model_arch = model_arch
        self.__first_run = False
        self.__current_epoch = 0

        self.__log = logger.Logger()

        if model_name is None:
            self.__log.info("No model name provided.")
            self.__select_model_without_name()
        else:
            self.__log.info("Model name provided.")
            self.__select_model_with_name(model_name)

        self.__create_model_first_run()
        self.__load_asset()
        self.__log.info(f"Model name choose: {self.__model_path.stem}, is first run ? : {self.__first_run}.")

    def __select_model_with_name(self, model_name: str) -> None:
        available_models = find_models(self.__models_dir, self.__model_arch)
        for key, value in available_models.items():
            decoded = decode_model_name(value)
            if decoded[1] == model_name:
                selected_model = [model for model in self.__models_dir.rglob(f"*{value}*.keras")][0]
                self.__model_path = selected_model
                self.__input_shape = decoded[2]
                self.__first_run = False
                self.__log.info("Found a match with model name.")
                return
        self.__log.info(f"No matching model name for : {model_name} --> new model.")
        self.__first_run = True
        self.__model_path = new_model_format(self.__model_arch, model_name, self.__input_shape)
        self.__model_path = self.get_workspace_path.joinpath(self.__model_path + ".keras")

    def __select_model_without_name(self) -> None:
        available_models = find_models(self.__models_dir, self.__model_arch)
        self.__model_path = choose_models(available_models, self.__models_dir)
        if self.__model_path is None:
            self.__first_run = True
        else:
            decoded = decode_model_name(self.__model_path.stem)
            self.__input_shape = decoded[2]

    def __create_model_first_run(self) -> None:
        if self.__first_run and self.__model_path is None:
            model_name = model_first_run_choose_name()
            model_full_name = new_model_format(self.__model_arch, model_name, self.__input_shape)
            self.__model_path = self.get_workspace_path.joinpath(model_full_name + ".h5")

    def __load_asset(self):
        self.__log.info("Loading asset...")
        pkl_checkpoint = self.__model_path.with_suffix('.pkl')
        if not pkl_checkpoint.exists():
            self.__log.info("No asset can be loaded -> first run.")
            return
        with open(pkl_checkpoint, 'rb') as f:
            d = pickle.load(f)
            self.__current_epoch = d['epoch']
            self.__optimizer = d['optimizer']
        self.__log.info("Asset loaded.")

    def compile(self, learning_rate=1e-5) -> None:
        if self.__optimizer is None:
            self.__log.info("Using new optimizer.")
            self.__model.compile(loss=self.__get_losses_metric__(),
                                 optimizer=optimizers.Adam(learning_rate=learning_rate),
                                 metrics=['accuracy'])
            return
        self.__log.info(f"Loading optimizer and previous state from {self.__optimizer}")
        self.__model.compile(loss=self.__get_losses_metric__(),
                             optimizer=optimizers.Adam.from_config(self.__optimizer),
                             metrics=['accuracy'])

    def show_summary(self) -> None:
        self.__model.summary(print_fn=self.__log.info)
        line_length = 65
        sep = ' : '
        info_key = ['Model name', 'Shape', 'Start epoch', 'Optimizers']
        info_value = [str(self.__model_path.stem), str(self.__input_shape), self.__current_epoch, str(self.__optimizer)]
        for idx, key in enumerate(info_key):
            value = str(info_value[idx])
            key_length = len(key)
            value_length = len(value)
            total_space = line_length - key_length - value_length - len(sep)
            left_space = total_space // 2
            right_space = total_space - left_space
            self.__log.info(f"{left_space * ' '}{key}{sep}{value}{right_space * ' '}")
        self.__log.info(f"{line_length * '='}")

    def __setup_model__(self):
        if self.__model_path and self.__model_path.exists():
            self.__log.info("Loading model...")
            self.__model = load_model(str(self.__model_path), compile=False)
            self.__log.info(" > Done")
            return
        self.__log.info("Creating model...")
        kwargs = {
            'include_top': True,
            'weights': None,
            'input_shape': self.get_input_shape,
            'classes': 2
        }
        self.__model = self.__get_architecture_class__()(**kwargs)
        self.__log.info(" > Done")

    @abstractmethod
    def __get_losses_metric__(self) -> any:
        pass

    @abstractmethod
    def __get_architecture_class__(self) -> any:
        pass

    @property
    def get_keras_model_path(self) -> Path:
        return self.__model_path

    @property
    def get_keras_model(self) -> Optional[Union[Model, Sequential]]:
        return self.__model

    @property
    def get_workspace_path(self) -> Path:
        return self.__models_dir.joinpath(self.__model_arch)

    @property
    def get_input_shape(self) -> Tuple[int, int, int]:
        return self.__input_shape

    @property
    def get_start_epoch(self):
        return self.__current_epoch

    @property
    def is_first_run(self):
        return self.__first_run

    @property
    def get_arch(self):
        return self.__model_arch

