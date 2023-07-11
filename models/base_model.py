import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import tensorflow.keras
import tensorflow.keras as keras

from log_io import logger
from log_io.logger import Logger


# Model name format : {models arch}_{model_name}_{dim: xy_channel}.*

class ModelBase(ABC):
    def __init__(self, models_dir: Path, model_arch: str, input_shape: Tuple[int, int, int], nb_epoch: int,
                 batch_size: int = 32, model_name: Optional[str] = None) -> None:
        super().__init__()

        self.__optimizer = None
        self.__input_shape = input_shape
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
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
                selected_model = [model for model in self.__models_dir.rglob(f"*{value}*")][0]
                self.__model_path = selected_model
                self.__input_shape = decoded[2]
                self.__log.info("Found a match with model name.")
                return
        self.__log.info(f"No matching model name for : {model_name} --> new model.")
        self.__first_run = True
        self.__model_path = new_model_format(self.__model_arch, model_name, self.__input_shape)
        self.__model_path = self.workspace_path.joinpath(self.__model_path + ".h5")

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
            self.__model_path = self.workspace_path.joinpath(model_full_name + ".h5")

    def __load_asset(self):
        self.__log.info("Loading asset...")
        pkl_checkpoint = self.keras_model_path.with_suffix('.pkl')
        if not pkl_checkpoint.exists():
            self.__log.info("No asset can be loaded -> first run.")
            return
        with open(pkl_checkpoint, 'rb') as f:
            d = pickle.load(f)
            self.__current_epoch = d['epoch']
            self.__optimizer = d['optimizer']
        self.__log.info("Asset loaded.")

    @abstractmethod
    def show_summary(self) -> None:
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

    @abstractmethod
    def compile(self) -> None:
        pass

    @abstractmethod
    def keras_model(self) -> Optional[Union[keras.Model, keras.Sequential]]:
        return None

    @property
    def keras_model_path(self) -> Path:
        return self.__model_path

    @property
    def workspace_path(self) -> Path:
        return self.__models_dir.joinpath(self.__model_arch)

    @property
    def input_shape(self) -> Tuple[int , int, int]:
        return self.__input_shape

    @property
    def models_dir(self) -> Union[Path, any]:
        return self.__models_dir

    @property
    def log(self):
        return self.__log

    @property
    def start_epoch(self):
        return self.__current_epoch

    @property
    def optimizer(self):
        return self.__optimizer

    def is_first_run(self):
        return self.__first_run


def new_model_format(model_arch: str, model_name: str, input_shape: Tuple[int, int, int]):
    return "_".join([model_arch, model_name, f"{input_shape[0]}_{input_shape[2]}"])


def model_name_from_path(model_path: Path) -> str:
    model_stem = model_path.stem
    arch, name, shape = decode_model_name(model_stem)
    model_name = new_model_format(arch, name, shape)
    return model_name


def decode_model_name(model_name_stem: str) -> Tuple[str, str, Tuple[int, int, int]]:
    split = model_name_stem.split("_")
    arch = split[0]
    str_split_dim = [split[-2], split[-1]]
    shape = (int(str_split_dim[0]), int(str_split_dim[0]), int(str_split_dim[1]))
    name = "_".join(split[1:-2])
    return arch, name, shape


def find_models(model_dir: Path, model_arch: str) -> Dict[int, str]:
    models = dict()
    curr_index = 0
    for name in model_dir.rglob(f"{model_arch}*"):
        if not name.exists() or name.is_dir() or name.suffix != ".h5":
            continue
        models[curr_index] = name.stem
        curr_index += 1
    return models


def choose_models(available_models: Dict[int, str], model_dir: Path) -> Optional[Path]:
    if not available_models:
        return None

    log = Logger()

    log.info(" -- Choose a Model --")
    for index, name in available_models.items():
        log.info(f"  -> [{index}]: {name}")

    default_choice = 0
    log.info(f"Enter your models index choice (default: {default_choice}), enter -1 to create a new model")
    user_input = input(" > ")

    try:
        choice = int(user_input)
    except ValueError as e:
        log.err(str(e))
        log.info(f"New models selected, name : {user_input}.")
        log.info("Choice not an integer. Using default choice.")
        log.err("Choice not an integer. Using default choice.")
        choice = default_choice

    if choice == -1:
        return None

    if choice < 0 or choice >= len(available_models):
        log.info("Choice out of range. Using default choice.")
        log.err("Choice out of range. Using default choice.")
        choice = default_choice

    selected_model = available_models[choice]
    selected_model = [model for model in model_dir.rglob(f"*{selected_model}*")][0]

    return selected_model


def model_first_run_choose_name() -> str:
    log = Logger()

    default_choice = "deepfake-detection"
    log.info(" -- First Time: Choose a model name --")
    log.info(f"Enter your model name (default: {default_choice}) ")
    user_input = input(" > ")
    return user_input
