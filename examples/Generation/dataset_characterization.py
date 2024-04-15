from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import plot_images, plot_hists

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'

figsize = (8, 8)
params = np.array([['chirp_mass', 'luminosity_distance'], ['ra', 'dec']])
figs = []
for seed in [0, 61]:
    dataset = load_rawsets(rawdat_dir, seeds2names(seed))
    dataset.change_parameter_name('d_L', to='luminosity_distance')
    indices = np.arange(12).reshape((3, 4))
    figs.append(plot_images(dataset, indices, figsize=figsize))
    # hist = plot_hists(dataset, params, figsize=figsize, bins=25)
    # hist.suptitle(f'dataset: {dataset.name}')
    # figs.append(hist)

plt.show()
