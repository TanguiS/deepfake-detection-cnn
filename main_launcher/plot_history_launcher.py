from pathlib import Path
from typing import Dict, Optional


def decode_plot_args(args: Dict[str, any]):
    kwargs = {
        'models_dir': Path(args['root_face_folder']),
        'model_arch': str(args['arch']),
        'model_name': args['model_name']
    }
    return kwargs


def launch_plot(models_dir: Path, model_arch: str, model_name: Optional[str]) -> None:
    from bench.plot_history import plot_training_results

    plot_training_results(models_dir, model_arch, model_name)
