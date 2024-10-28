import torch
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pesummary.gw.plots.latex_labels import GWlatex_labels

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image


def add_identity(axes, *line_args, **line_kwargs):
    # https://stackoverflow.com/questions/22104256/does-matplotlib-have-a-function-for-drawing-diagonal-lines-in-axis-coordinates
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


n = 0
m = 0
letter = ''
files_dir = Path('/mnt/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'Originals' #'mid range' / 'LIGO-O2'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 5. 14 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'

flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv5.{n}.{m}{letter}.pt')
flow0.rename(f'Spv5.{n}.{m}{letter}')
flow0.eval()

zero_pad = 3 #4
seed = 999 # range(990, 1000)
# event = f'{seed:0{zero_pad}}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed, zero_pad=zero_pad))
dataset.change_parameter_name('d_L', to='luminosity_distance')
# trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')
#
# # n = None
# #
# sset = flow0.sample_set(3000, trainset[:][:], name=f'flow {flow0.name}')
# # 50+ minutes... for 999 @ 10_000
# # 2.5+ hours for 990-999 @ 3_000
# medians, truths = {}, {}
#
# for param in flow0.param_list:
#     # print(param)
#     # print()
#     medians[param] = []
#     truths[param] = []
#     for sdict in sset.values():
#         medians[param].append(np.median(sdict[param]).item())
#         truths[param].append(sdict.truth[param])
#         # print(f'{np.median(sdict[param])} vs {sdict.truth[param]}')
#     # print('\n'*2)
#
# # from pprint import pprint
# # pprint(medians)
#
# torch.save((medians, truths), f'assessment_{flow0.name}_{seed}.pt')
_medians, _truths = torch.load(f'assessment_{flow0.name}_{seed}.pt')
flow0.rename(f'GP14')

snr = dataset['SNR'].apply(lambda x: np.sqrt(np.sum([val**2 for val in x.values()]))).to_numpy()  # Originals
# snr = dataset['parameters'].apply(lambda x: x['_snr']).to_numpy()  # New pipe-convention
# print(np.min(snr))  # min = 6.5 for originals, 5 for new pipe
# snr_min = 5
# Same ranges and bins for consistency, can be changed
hist_range = {}
_bins = {}
for snr_min in [5, 20]: #[10, 15, 20, 25, 30, 40, 50]:
    indexes = np.where(snr > snr_min)[0]
    # [(m.pop(index), t.pop(index)) for index in sorted(indexes, reverse=True)
    #  for m, t in zip(medians.values(), truths.values())]
    medians = {key: [val[index] for index in indexes] for key, val in _medians.items()}
    truths = {key: [val[index] for index in indexes] for key, val in _truths.items()}

    # medians = {key: np.zeros_like(val) for key, val in medians.items()}
    # ax: plt.Axes = plt.Axes()  # Type hinting trick
    fig: plt.Figure
    fig, axs = plt.subplots(2, len(flow0.param_list) // 2, figsize=(20, 10))
    bins = 40
    norm = mpl.colors.Normalize(vmin=0, vmax=20.0)
    for i, ax in enumerate(axs.flatten()):
        key = flow0.param_list[i]
        # if key == 'luminosity_distance':
        #     ax.set_xscale('log')
        #     ax.set_yscale('log')
        #     _bins = np.logspace(start=np.log10(100), stop=np.log10(4000), num=20)
        # else:
        #     _bins = bins

        # Set limits on the first frame
        if snr_min == 5:
            # Dynamically adjust bins to desired format even if estimations are compressed
            t_range = (np.max(truths[key]) - np.min(truths[key]))
            m_range = (np.max(medians[key]) - np.min(medians[key]))
            _bins[key] = int(bins * np.min([t_range / m_range, 2]))
            # Make histogram range identical
            hist_range[key] = np.array([[np.min(truths[key]), np.max(truths[key])] for _ in range(2)])
        H, xedges, yedges = np.histogram2d(medians[key],
                                           truths[key],
                                           bins=_bins[key],
                                           range=hist_range[key])
        # print(np.max(H))
        ax.set_box_aspect(1)
        # Taking care of conventions x->y, y->x
        pcolor = ax.pcolormesh(yedges, xedges, H, cmap='turbo', norm=norm)
        ax.set_title(GWlatex_labels[key])
        # ax.set_yticks(ax.get_xticks())
        add_identity(ax, color='w', ls=':')
        # ax.plot([0, 1], [0, 1], 'k--', transform=ax.transAxes)
        # if i in [len(flow0.param_list)//2-1, 2*len(flow0.param_list)//2-1]:
        if i == len(flow0.param_list) // 2:
            ax.set_ylabel('estimated medians', fontsize='large')
            ax.set_xlabel('injection values', fontsize='large')
    fig.colorbar(pcolor, ax=axs, orientation='horizontal', fraction=.1, pad=-0.32, aspect=50, extend='max')
    fig.suptitle(f'{flow0.name} assessment on validation set {seed} for SNR > {snr_min}', y=0.78, fontsize='xx-large')
    fig.tight_layout(h_pad=2.2)
    plt.savefig(f'assessment/new names/{flow0.name}_{seed}_{snr_min}.png', bbox_inches='tight')

