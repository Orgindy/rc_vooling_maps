import os
import matplotlib.pyplot as plt

# Use modern seaborn styling for clarity
plt.style.use('seaborn-v0_8-whitegrid')


def apply_standard_plot_style(ax, title=None, xlabel=None, ylabel=None, grid=True):
    """
    Apply consistent styling to a matplotlib axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to style.
    title : str, optional
        Title for the plot.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    grid : bool, optional
        Whether to show a grid (default: True).
    """
    ax.tick_params(labelsize=12)
    ax.set_xlabel(xlabel or "", fontsize=14)
    ax.set_ylabel(ylabel or "", fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    if grid:
        ax.grid(True, linestyle='--', alpha=0.5)


def save_figure(fig, filename, dpi=300, folder='results/maps', transparent=False):
    """
    Save a matplotlib figure with standard settings.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    filename : str
        The file name (with or without extension).
    dpi : int, optional
        Resolution in dots per inch.
    folder : str, optional
        Destination folder.
    transparent : bool, optional
        Save with transparent background (for overlays).
    """
    os.makedirs(folder, exist_ok=True)

    # Ensure filename has an extension
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
        filename += '.png'

    fig.tight_layout()
    output_path = os.path.join(folder, filename)
    fig.savefig(output_path, dpi=dpi, transparent=transparent)
    plt.close(fig)
    print(f"✅ Figure saved: {output_path}")
