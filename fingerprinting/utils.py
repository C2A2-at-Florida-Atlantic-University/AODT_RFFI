import numpy as np
import matplotlib.pyplot as plt

def generate_grid_node_ids():
    ids = {}
    coordinates = {}
    node_i = 0
    for i in np.arange(1, 21):
        for j in np.arange(1, 21):
            ids[str(i) + "-" + str(j)] = node_i
            node_i = node_i + 1
            coordinates[node_i] = (i, j)
    return ids, coordinates

def apply_ieee_style():
    plt.style.use('default')
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 80,
        "savefig.dpi": 300,
        "text.usetex": False,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.5,
        "lines.linewidth": 0.5,

        # Grayscale settings
        "image.cmap": "gray",
        "axes.facecolor": 'white',
        "figure.facecolor": 'white',
        "figure.edgecolor": 'white',
        "savefig.facecolor": 'white',
        "savefig.edgecolor": 'white',
        "grid.color": '0.8',
        "text.color": 'black',
        "axes.edgecolor": 'black',
        "axes.labelcolor": 'black',
        "xtick.color": 'black',
        "ytick.color": 'black'
    })