from pathlib import Path
from typing import Tuple, Dict, Optional

from log_io.logger import Logger


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
    str_split_dim = [split[2], split[3]]
    shape = (int(str_split_dim[0]), int(str_split_dim[0]), int(str_split_dim[1]))
    name = split[1]
    return arch, name, shape


def find_models(model_dir: Path, model_arch: str) -> Dict[int, str]:
    models = dict()
    curr_index = 0
    for name in model_dir.rglob(f"{model_arch}*"):
        if name.suffix == ".h5" or name.suffix == ".tf" or name.suffix == ".keras":
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
