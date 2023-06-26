import datetime
from pathlib import Path
from threading import Lock
from typing import Optional


def message_format(message):
    current_date = datetime.datetime.now().strftime("[%d-%m-%Y-%H-%M-%S]")
    log_message = f"{current_date}: {message}"
    return log_message


def print_log(message: str) -> str:
    log_message = message_format(message)
    print(log_message)
    return log_message


def write_log(file: Optional[Path], log_message: str) -> None:
    if file:
        with open(file, "a") as f:
            f.write(log_message + "\n")


class SingletonMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Logger(metaclass=SingletonMeta):
    def __init__(self, path_to_log_dir: Optional[Path] = None):
        self.__log_info = None
        self.__log_error = None
        self.__dir = None
        if path_to_log_dir and path_to_log_dir.exists() and path_to_log_dir.is_dir():
            self.__root_log = path_to_log_dir.joinpath("log")
            self.__root_log.mkdir(exist_ok=True)
            log_file_name_pattern = f"{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
            self.__dir = self.__root_log.joinpath(log_file_name_pattern)
            self.__dir.mkdir(exist_ok=True)
            self.__log_info = self.__dir.joinpath(f"log_info.txt")
            self.__log_error = self.__dir.joinpath(f"log_error.txt")
            self.__log_error.touch()
            self.__log_info.touch()

    def info(self, message: str) -> None:
        log_message = print_log(message)
        write_log(self.__log_info, log_message)

    def info_file(self, message: str) -> None:
        write_log(self.__log_info, message_format(message))

    def err(self, message: str) -> None:
        log_message = print_log(message)
        write_log(self.__log_error, log_message)
