from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_training_results(train_csv_path: Path, val_csv_path: Path) -> None:
    train_data = pd.read_csv(train_csv_path, delimiter=';')
    val_data = pd.read_csv(val_csv_path, delimiter=';')

    # Calculate average train loss and accuracy for each epoch
    train_avg_loss = train_data.groupby('epoch')['loss'].mean()
    train_avg_acc = train_data.groupby('epoch')['acc'].mean()

    # Set the style of the plot
    plt.style.use('seaborn-whitegrid')

    # Create a figure and axes
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Define color gradients
    train_loss_color = '#FF0000'  # Red
    val_loss_color = '#FF8C00'  # Orange
    train_acc_color = '#008000'  # Green
    val_acc_color = '#00CED1'  # Blueish

    # Plot loss
    ax1 = axes[0]
    # ax1.plot(train_data['scaled_batch'], train_data['loss'], color=train_loss_color, alpha=0.1)
    ax1.plot(val_data['epoch'], val_data['val_loss'], label='Validation Loss', color=val_loss_color)
    # ax1.scatter(train_data['scaled_batch'], train_data['loss'], color=train_loss_color, s=5, alpha=0.1)
    ax1.scatter(val_data['epoch'], val_data['val_loss'], color=val_loss_color, s=10)
    ax1.plot(train_avg_loss.index, train_avg_loss, label='Training Loss', color=train_loss_color)
    ax1.scatter(train_avg_loss.index, train_avg_loss, color=train_loss_color, s=10)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Value')
    ax1.set_title('Losses Metrics')
    ax1.set_ylim(0, 1)
    ax1.legend()

    # Plot accuracy
    ax2 = axes[1]
    ax2.plot(val_data['epoch'], val_data['val_acc'], label='Validation Accuracy', color=val_acc_color)
    ax2.scatter(val_data['epoch'], val_data['val_acc'], color=val_acc_color, s=10)
    ax2.plot(train_avg_acc.index, train_avg_acc, label='Training Accuracy', color=train_acc_color)
    ax2.scatter(train_avg_acc.index, train_avg_acc, color=train_acc_color, s=10)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Value')
    ax2.set_title('Accuracies Metrics')
    ax2.set_ylim(0, 1)
    ax2.legend()

    # Add grid lines to the plot
    ax1.grid(True)
    ax2.grid(True)

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
