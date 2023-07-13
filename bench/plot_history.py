from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

from models import base_model


def plot_training_results(train_csv_path: Path, val_csv_path: Path) -> None:
    arch, _, shape = base_model.decode_model_name(train_csv_path.stem)
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
