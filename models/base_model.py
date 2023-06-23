import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Optional, Union

import keras
from keras.optimizers import Optimizer

from log_io import logger
from log_io.logger import Logger


# Model name format : {models arch}_{model_name}_{dim: xy_channel}_{reached_nb_epoch}.*

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
            self.__select_model_without_name()
        else:
            self.__select_model_with_name(model_name)

        self.__create_model_first_run()
        self.__log.info(f"Model name choose: {self.__model_full_name}, is first run ? : {self.__first_run}.")

    def __select_model_with_name(self, model_name: str) -> None:
        available_models = find_models(self.__models_dir, self.__model_arch)
        for key, value in available_models.items():
            decoded = decode_model_name(value)
            if decoded[1] == model_name:
                selected_model = [model for model in self.__models_dir.rglob(f"*{value}*")][0]
                self.__model_path = selected_model
                self.__model_full_name = model_name_from_path(self.__model_path)
                pass
        self.__first_run = True

    def __select_model_without_name(self) -> None:
        available_models = find_models(self.__models_dir, self.__model_arch)
        self.__model_path = choose_models(available_models, self.__models_dir)
        if self.__model_path is None:
            self.__first_run = True
        else:
            self.__model_full_name = model_name_from_path(self.__model_path)

    def __create_model_first_run(self) -> None:
        if self.__first_run:
            model_name = model_first_run_choose_name()
            self.__model_full_name = new_model_format(self.__model_arch, model_name, self.__input_shape)
            self.__model_path = None

    @abstractmethod
    def show_summary(self) -> None:
        pass

    @abstractmethod
    def compile(self) -> None:
        pkl_checkpoint = self.__model_path.joinpath(self.__model_full_name + ".pkl")
        if pkl_checkpoint is None:
            pass
        with open(pkl_checkpoint, 'rb') as f:
            d = pickle.load(f)
            self.__current_epoch = d['epoch']
            self.__optimizer = d['optimizer']

    @abstractmethod
    def model(self) -> Optional[Union[keras.Model, keras.Sequential]]:
        return None

    @property
    def model_hdf5_file_path(self) -> Path:
        return self.__model_path

    @property
    def workspace(self) -> Path:
        return self.__models_dir.joinpath(self.__model_arch)

    @property
    def shape(self) -> Tuple[int , int, int]:
        return self.__input_shape

    @property
    def models_dir(self) -> Union[Path, any]:
        return self.__models_dir

    @property
    def model_name_with_hdf5(self) -> str:
        return self.__model_full_name + ".hdf5"

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
    arch, name, shape, nb_epoch = decode_model_name(model_stem)
    model_name = new_model_format(arch, name, shape)
    return model_name


def decode_model_name(model_name_stem: str) -> Tuple[str, str, Tuple[int, int, int], int]:
    split = model_name_stem.split("_")
    arch = split[0]
    nb_epoch = int(split[-1])
    str_split_dim = split[-3:-2]
    shape = (int(str_split_dim[0]), int(str_split_dim[0]), int(str_split_dim[1]))
    name = "_".join(split[1:-2])
    return arch, name, shape, nb_epoch


def find_models(model_dir: Path, model_arch: str) -> Dict[int, str]:
    models = dict()
    for index, name in enumerate(model_dir.rglob(f"{model_arch}*")):
        if name.is_dir():
            continue
        models[index] = name.stem
    return models


def choose_models(available_models: Dict[int, str], model_dir: Path) -> Optional[Path]:
    if not available_models:
        return None

    logger = Logger.get_instance()

    logger.info(" -- Choose a Model --")
    for index, name in available_models.items():
        logger.info(f"  -> [{index}]: {name}")

    default_choice = 0
    logger.info(f"Enter your models index choice (default: {default_choice})")
    user_input = input()

    try:
        choice = int(user_input)
    except ValueError as e:
        logger.err(e)
        logger.info(f"New models selected, name : {user_input}.")
        choice = user_input

    if choice < 0 or choice >= len(available_models):
        logger.info("Choice out of range. Using default choice.")
        logger.err("Choice out of range. Using default choice.")
        choice = default_choice

    selected_model = available_models[choice]
    selected_model = [model for model in model_dir.rglob(f"*{selected_model}*")][0]

    return selected_model


def model_first_run_choose_name() -> str:
    logger = Logger()

    default_choice = "deepfake-detection"
    logger.info(" -- First Time: Choose a model name --")
    logger.info(f"Enter your model name (default: {default_choice}) ")
    user_input = input(" > ")
    return user_input
