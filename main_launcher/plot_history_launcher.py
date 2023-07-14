from pathlib import Path
from typing import Dict


def decode_plot_args(args: Dict[str, any]):
    kwargs = {
        'train_csv_path': Path(args['train_history_csv_path']),
        'val_csv_path': Path(args['validation_history_csv_path'])
    }
    return kwargs


def launch_plot(train_csv_path: Path, val_csv_path: Path) -> None:
    from bench.plot_history import plot_training_results
    plot_training_results(train_csv_path, val_csv_path)
