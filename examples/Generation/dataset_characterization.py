from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_rawsets_pool, seeds2names
from dtempest.gw.conversion import plot_hists

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'mid range' / 'LIGO-O2'

figsize = (12, 8)
params = np.transpose(np.array([['chirp_mass', 'mass_ratio'],
                                ['luminosity_distance', 'theta_jn'],
                                ['ra', 'dec']]))
figs = []
for seed in [range(240),]:
    dataset = load_rawsets_pool(rawdat_dir, seeds2names(seed, zero_pad=4)) #, processes=12)
    from pprint import pprint
    pprint(dataset.metadata)
    # pprint(dataset.loc['LIGO-O2.0000.00000']['parameters'])
    # dataset.change_parameter_name('d_L', to='luminosity_distance')
    # indices = np.arange(12, 24).reshape((3, 4))
    # figs.append(plot_images(dataset, indices, figsize=figsize))
    hist = plot_hists(dataset, params, figsize=figsize, bins=50, title='')

    # ax = hist.get_axes()[-1]
    # ax.clear()
    # ax.hist([inj['parameters']['_snr'] for inj in dataset if inj['parameters']['_snr'] < 40], bins=50)

    # hist.suptitle(f'dataset: {dataset.name}')
    hist.suptitle('Parameter distributions on chunks 0 through 239 of the LIGO-O2 dataset')
    plt.tight_layout()
    # plt.savefig('Distribution_0_through_49.png')
    # figs.append(hist)

plt.show()
