from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import plot_images, plot_hists

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'

figsize = (12, 8)
params = np.transpose(np.array([['chirp_mass', 'mass_ratio'], ['luminosity_distance', 'theta_jn'], ['ra', 'dec']]))
figs = []
for seed in [range(50)]:
    dataset = load_rawsets(rawdat_dir, seeds2names(seed))
    dataset.change_parameter_name('d_L', to='luminosity_distance')
    # indices = np.arange(12).reshape((3, 4))
    # figs.append(plot_images(dataset, indices, figsize=figsize))
    hist = plot_hists(dataset, params, figsize=figsize, bins=50, title='')
    # hist.suptitle(f'dataset: {dataset.name}')
    hist.suptitle('Parameter distributions on datasets 0 through 49')
    plt.tight_layout()
    # plt.savefig('Distribution_0_through_49.png')
    # figs.append(hist)

plt.show()
