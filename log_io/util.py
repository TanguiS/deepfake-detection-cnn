import datetime
from pathlib import Path
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
