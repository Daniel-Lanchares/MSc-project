from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_rawsets_pool, seeds2names
from dtempest.gw.conversion import plot_hist_ax, extract_parameters, extract_SNR

files_dir = Path('/mnt/easystore/Daniel/MSc-files')
rawdat_dir = (files_dir / 'Raw Datasets' / 'mid range' / 'LIGO-O2',
              files_dir / 'Raw Datasets' / 'Originals')
# rawdat_dir = (files_dir / 'Raw Datasets' / 'mid range' / 'LIGO-O2',
#               files_dir / 'Raw Datasets' / 'mid range' / 'GW170823')

figsize = (15, 9)
params = np.transpose(np.array([['chirp_mass', 'mass_ratio', 'luminosity_distance'],
                                ['a_1', 'tilt_1', 'phi_12'],
                                ['a_2', 'tilt_2', 'phi_jl'],
                                ['theta_jn', 'psi','phase'],
                                ['ra', 'dec', '_snr']]))
padding = (4, 3)
colors = ('tab:orange', 'tab:blue')
# figs = []
fig, axs = plt.subplots(*params.shape, figsize=figsize)
a, b = 250, 50  # 10, 2
for i, seed in enumerate([range(a), range(b)]):
    dataset = load_rawsets_pool(rawdat_dir[i], seeds2names(seed, zero_pad=padding[i])) #, processes=12)
    if i==1:
        dataset.change_parameter_name('d_L', to='luminosity_distance')
        alpha = 0.85
    else:
        alpha = 1
    # from pprint import pprint
    # pprint(dataset.metadata)
    # pprint(dataset.loc['LIGO-O2.0000.00000']['parameters'])
    # dataset.change_parameter_name('d_L', to='luminosity_distance')
    # indices = np.arange(12, 24).reshape((3, 4))
    # figs.append(plot_images(dataset, indices, figsize=figsize))
    # hist = plot_hists(dataset, params, fig=fig, bins=50, title='', color=colors[i])
    for ax, param in zip(axs.flatten(), params.flatten()):
        if param == '_snr':
            if i == 1:
                data = extract_SNR(dataset, ['L1', 'H1', 'V1'])
                # print(data)
                ax.hist(data, bins=30, color=colors[i], alpha=alpha)
            else:
                plot_hist_ax(dataset, param, ax, bins=12, title='', color=colors[i], alpha=alpha)
            ax.set_yscale('log')
            # ax.set_title('SNR histogram (ln(events))')
            ax.set_xlabel('SNR (log(events))')
        else:
            plot_hist_ax(dataset, param, ax, bins=30, title='', color=colors[i], alpha=alpha)

    # ax = fig.get_axes()[-1]
    # ax.clear()
    # ax.hist([inj['parameters']['_snr'] for inj in dataset if inj['parameters']['_snr'] < 40], bins=50)

    # hist.suptitle(f'dataset: {dataset.name}')
    # hist.suptitle('Parameter distributions on chunks 0 through 239 of the LIGO-O2 dataset')
    fig.suptitle('Parameter distributions: GP dataset (blue) vs LH dataset (orange)')
    plt.tight_layout()
    # plt.savefig('Distribution_0_through_49.png')
    # figs.append(hist)
    del dataset

plt.show()
