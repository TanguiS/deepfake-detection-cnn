from pathlib import Path
from typing import Tuple, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

import models.util
from models import util


def find_history_csv(models_dir: Path, model_arch: str, model_name: Optional[str]) -> Tuple[Path, Path]:
    model_stem = get_model(model_arch, model_name, models_dir)
    return models_dir.joinpath(model_arch).joinpath(f"{model_stem}_train_history.csv"), \
           models_dir.joinpath(model_arch).joinpath(f"{model_stem}_val_history.csv")


def get_model(model_arch, model_name, models_dir):
    available_models = util.find_models(models_dir, model_arch)
    if len(available_models) == 0:
        raise ValueError
    model_stem = None
    if model_name is None:
        model_stem = util.choose_models(available_models, models_dir).stem
    else:
        for key, value in available_models.items():
            if util.decode_model_name(value)[1] == model_name:
                model_stem = value
                break
    if model_stem is None:
        raise ValueError
    return model_stem


def plot_training_results(models_dir: Path, model_arch: str, model_name: Optional[str]) -> None:
    train_csv_path, val_csv_path = find_history_csv(models_dir, model_arch, model_name)
    arch, _, shape = models.util.decode_model_name(train_csv_path.stem)
    plot_train_history(train_csv_path, arch, shape)
    plot_val_history(val_csv_path, arch, shape)


def plot_train_history(train_csv_path: Path, arch: str, shape: Tuple[int, int, int]):
    train_data = pd.read_csv(train_csv_path, delimiter=';')

    plt.style.use('seaborn-whitegrid')
    gs = gridspec.GridSpec(2, 4, wspace=1., hspace=.35)
    fig: plt.Figure = plt.figure(figsize=(8, 8))

    train_time = [train_data['time (s)'][0]]
    for i in range(1, len(train_data['time (s)'])):
        train_time.append(train_time[-1] + train_data['time (s)'][i])

    ax1: plt.Axes = fig.add_subplot(gs[0, 0:4])
    ax1.plot(train_time, train_data['scaled_batch'], linewidth=3, color='orange')
    ax1.set_title('Training Time (s)', fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Epoch", fontsize=16)

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(train_data['scaled_batch'], train_data['loss'], linewidth=3, color='royalblue')
    ax2.set_title('Training Loss', fontsize=16)
    ax2.set_xlabel("Epoch", fontsize=16)
    ax2.set_ylabel("Loss", fontsize=16)

    ax3 = fig.add_subplot(gs[1, 2:])
    ax3.plot(train_data['scaled_batch'], train_data['accuracy'], linewidth=3, color='mediumseagreen')
    ax3.set_title('Training Accuracy', fontsize=16)
    ax3.set_xlabel("Epoch", fontsize=16)
    ax3.set_ylabel("Accuracy", fontsize=16)

    fig.suptitle(f'Training History - {arch} / {shape}', fontsize=24)
    plt.show()


def plot_val_history(val_csv_path: Path, arch: str, shape: Tuple[int, int, int]):
    val_data = pd.read_csv(val_csv_path, delimiter=';')

    plt.style.use('seaborn-whitegrid')
    gs = gridspec.GridSpec(2, 2, wspace=.35, hspace=.35)
    fig: plt.Figure = plt.figure(figsize=(8, 8))

    train_time = [val_data['time (s)'][0]]
    for i in range(1, len(val_data['time (s)'])):
        train_time.append(train_time[-1] + val_data['time (s)'][i])

    ax1: plt.Axes = fig.add_subplot(gs[0, 0])
    ax1.plot(train_time, val_data['epoch'], linewidth=3, color='orange')
    ax1.set_title('Training Time (s)', fontsize=16)
    ax1.set_xlabel("Time (s)", fontsize=16)
    ax1.set_ylabel("Epoch", fontsize=16)

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(val_data['epoch'], val_data['lr'], linewidth=3, color='violet')
    ax1.set_title('Learning Rate', fontsize=16)
    ax1.set_xlabel("Epoch", fontsize=16)
    ax1.set_ylabel("Learning rate", fontsize=16)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(val_data['epoch'], val_data['val_loss'], linewidth=3, color='royalblue')
    ax3.set_title('Validation Loss', fontsize=16)
    ax3.set_xlabel("Epoch", fontsize=16)
    ax3.set_ylabel("Loss", fontsize=16)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(val_data['epoch'], val_data['val_accuracy'], linewidth=3, color='mediumseagreen')
    ax4.set_title('Validation Accuracy', fontsize=16)
    ax4.set_xlabel("Epoch", fontsize=16)
    ax4.set_ylabel("Accuracy", fontsize=16)

    fig.suptitle(f'Validation History - {arch} / {shape}', fontsize=24)
    plt.show()
