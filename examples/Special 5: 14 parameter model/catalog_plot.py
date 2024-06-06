from pathlib import Path
import matplotlib.pyplot as plt
import json
import numpy as np
from copy import copy

# from dtempest.core.common_utils import redraw_legend
#
# from dtempest.gw import CBCEstimator
# from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict
# from dtempest.gw.conversion import plot_image
# from dtempest.gw.catalog import Catalog, full_names
#
# from scipy import stats
# # from pesummary.utils.bounded_1d_kde import bounded_1d_kde
# from pesummary.gw.conversions import convert
from pesummary.gw.plots.latex_labels import GWlatex_labels

n = 2
m = 0
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 5. 14 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_1 = files_dir / 'GWTC-1 Samples'
catalog_2 = files_dir / 'GWTC-2.1 Samples'
catalog_3 = files_dir / 'GWTC-3 Samples'


# flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv5.{n}.{m}{letter}.pt')
# flow0.eval()
class flow0:
    # Placeholder for an actual model
    name = f'Spv5.{n}.{m}{letter}'


estimation_dir = files_dir / 'Estimation Data' / flow0.name
corner_dir = estimation_dir / 'Corners'

cat = 'gwtc-1'

with open(estimation_dir / f'{flow0.name}_{cat.upper()}.json', 'r') as fp:
    dtpst_data = json.load(fp)

with open(estimation_dir / f'{cat.upper()}.json', 'r') as fp:
    gwtc_data = json.load(fp)

names = np.array(list(dtpst_data.keys()))
print(names)


def for_gwtc1():
    if cat != 'gwtc-1':
        raise ValueError(f'This function is for GWTC-1 only, not for {cat.upper()}')

    offset = 3
    nparams = 10

    fig, axs = plt.subplots(2, nparams // 2, figsize=(20, 15), sharey='all')
    fig.get_axes()[0].set_yticks(np.arange(len(names)) + offset, names)
    axs = axs.flatten()

    gwtc_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#0072C1',

        'label': rf'{cat.upper()} 1$\sigma$ range'
    }

    dtpst_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#b30909',

        'label': rf'{flow0.name} 1$\sigma$ range'
    }

    for i, (event, evt_reference) in enumerate(gwtc_data.items()):
        for j, (param, reference) in enumerate(evt_reference.items()):
            gwtc_kwargs_copy = copy(gwtc_kwargs)
            dtpst_kwargs_copy = copy(dtpst_kwargs)
            if j == nparams:
                break
            if i != len(names) - 1 or j != nparams - 1:
                del gwtc_kwargs_copy['label'], dtpst_kwargs_copy['label']

            median, upper, lower = reference['median'], reference['upper'], reference['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **gwtc_kwargs_copy)
            predict = dtpst_data[event][param]
            median, upper, lower = predict['median'], predict['upper'], predict['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **dtpst_kwargs_copy)
            axs[j].set_xlabel(GWlatex_labels[param])

            if event == 'GW170817':
                axs[j].axhspan(i + offset - 0.25, i + offset + 0.25, color='tab:red', alpha=0.4)

    fig.legend(mode='expand', loc='upper center', ncols=2, numpoints=1,
               bbox_to_anchor=(0.12, 0.87, 0.785, 0.05), borderpad=0.7, )
    # title=f'{flow0.name} analysis of the {cat.upper()} catalog',)
    fig.savefig(estimation_dir / f'full_{cat.upper()}_{flow0.name}.png', bbox_inches='tight')
    # plt.show()


def for_gwtc2():
    if cat != 'gwtc-2.1':
        raise ValueError(f'This function is for GWTC-2.1 only, not for {cat.upper()}')

    offset = 3
    nparams = 14

    fig, axs = plt.subplots(2, nparams // 2, figsize=(30, 65), sharey='all')
    fig.get_axes()[0].set_yticks(np.arange(len(names)) + offset, names)
    axs = axs.flatten()

    gwtc_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#0072C1',

        'label': rf'{cat.upper()} 1$\sigma$ range'
    }

    dtpst_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#b30909',

        'label': rf'{flow0.name} 1$\sigma$ range'
    }

    for i, (event, evt_reference) in enumerate(gwtc_data.items()):
        for j, (param, reference) in enumerate(evt_reference.items()):
            gwtc_kwargs_copy = copy(gwtc_kwargs)
            dtpst_kwargs_copy = copy(dtpst_kwargs)
            if j == nparams:
                break
            if i != len(names) - 1 or j != nparams - 1:
                del gwtc_kwargs_copy['label'], dtpst_kwargs_copy['label']

            median, upper, lower = reference['median'], reference['upper'], reference['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **gwtc_kwargs_copy)
            predict = dtpst_data[event][param]
            median, upper, lower = predict['median'], predict['upper'], predict['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **dtpst_kwargs_copy)
            axs[j].set_xlabel(GWlatex_labels[param])

            if event in ['GW190425_081805', 'GW190814_211039']:
                axs[j].axhspan(i + offset - 0.25, i + offset + 0.25, color='tab:red', alpha=0.4)

    axs[8].set_xlim(right=8500)
    fig.legend(mode='expand', loc='upper center', ncols=2, numpoints=1,
               bbox_to_anchor=(0.122, 0.8465, 0.7812, 0.05), borderpad=0.7, )
    # title=f'{flow0.name} analysis of the {cat.upper()} catalog',)
    fig.savefig(estimation_dir / f'full_{cat.upper()}_{flow0.name}_large.png', bbox_inches='tight')
    # plt.show()


def for_gwtc3():
    if cat != 'gwtc-3':
        raise ValueError(f'This function is for GWTC-3 only, not for {cat.upper()}')

    offset = 3
    nparams = 14

    fig, axs = plt.subplots(2, nparams // 2, figsize=(30, 65), sharey='all')
    fig.get_axes()[0].set_yticks(np.arange(len(names)) + offset, names)
    axs = axs.flatten()

    gwtc_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#0072C1',

        'label': rf'{cat.upper()} 1$\sigma$ range'
    }

    dtpst_kwargs = {
        'capsize': 5,
        'elinewidth': 2,
        'marker': 's',
        'color': '#b30909',

        'label': rf'{flow0.name} 1$\sigma$ range'
    }

    for i, (event, evt_reference) in enumerate(gwtc_data.items()):
        for j, (param, reference) in enumerate(evt_reference.items()):
            gwtc_kwargs_copy = copy(gwtc_kwargs)
            dtpst_kwargs_copy = copy(dtpst_kwargs)
            if j == nparams:
                break
            if i != len(names) - 1 or j != nparams - 1:
                del gwtc_kwargs_copy['label'], dtpst_kwargs_copy['label']

            median, upper, lower = reference['median'], reference['upper'], reference['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **gwtc_kwargs_copy)
            predict = dtpst_data[event][param]
            median, upper, lower = predict['median'], predict['upper'], predict['lower']
            axs[j].errorbar(median, i + offset, xerr=np.array([[lower], [upper]]), **dtpst_kwargs_copy)
            axs[j].set_xlabel(GWlatex_labels[param])

            if event in ['GW200115_042309',]:
                axs[j].axhspan(i + offset - 0.25, i + offset + 0.25, color='tab:red', alpha=0.4)

    axs[8].set_xlim(right=8500)
    fig.legend(mode='expand', loc='upper center', ncols=2, numpoints=1,
               bbox_to_anchor=(0.122, 0.8465, 0.7812, 0.05), borderpad=0.7, )
    # title=f'{flow0.name} analysis of the {cat.upper()} catalog',)
    # fig.savefig(estimation_dir / f'full_{cat.upper()}_{flow0.name}_large.png', bbox_inches='tight')
    # plt.show()


for_gwtc1()
# for_gwtc2()
# for_gwtc3()
