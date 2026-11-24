# utils/visualization.py
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib
import os
# Find the folder containing the current file (config.py)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level -> init/, then into params/
SAVE_DIR = os.path.join(MODEL_DIR, "..", "params")
os.makedirs(SAVE_DIR, exist_ok=True)
import torch
import numpy as np

matplotlib.use('Agg')


def producing_lon_lat():
    """Generate longitude/latitude grid."""
    lat_vector = torch.linspace(-80 + 1/12, -80 + 2040 * 1 / 12, steps=2040)
    lon_vector = torch.linspace(-180, -180 + (4320 - 1) * 1 / 12, steps=4320)
    return lat_vector, lon_vector


def plotgraph(u, title='testing', save_path=True):
    """Plot a 2D image."""
    plt.figure(figsize=(10, 8))
    plt.imshow(u.cpu().numpy() if torch.is_tensor(u) else u,
               cmap='viridis', origin='lower')
    plt.colorbar(label='Value')
    plt.title(title)
    plt.xlabel('Longitude index')
    plt.ylabel('Latitude index')

    if save_path:
        plt.savefig(f'../fig/{title}.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_loss_curves(train_loss_history, val_loss_history, epoch_numbers, title):
    """Plot the training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_numbers, train_loss_history, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epoch_numbers, val_loss_history, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save the figure
    os.makedirs('./plot', exist_ok=True)
    plt.savefig(f'./plot/{title}_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save the loss data
    loss_data = {
        'epochs': epoch_numbers,
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    }
    torch.save(loss_data, os.path.join(SAVE_DIR, f'../plot/{title}_loss_data.pth'))


def get_norm_for_plotting(concatenated):
    """Get the normalization object used for plotting."""
    vmin = concatenated.min().item()
    vmax = concatenated.max().item()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    return norm
