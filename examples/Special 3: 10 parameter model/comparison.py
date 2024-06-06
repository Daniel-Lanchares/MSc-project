from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict

from dtempest.gw.conversion import plot_image
from dtempest.gw.catalog import Merger

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.gw.conversions import convert

'''

'''
n = 0
m = 0
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 3. 10 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_1 = files_dir / 'GWTC-1 Samples'
catalog_2 = files_dir / 'GWTC-2.1 Samples'
catalog_3 = files_dir / 'GWTC-3 Samples'

flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv3.{n}.{m}{letter}.pt')
flow0.rename(f'Spv3.{n}.{m}{letter}')
flow0.eval()
flow1 = CBCEstimator.load_from_file(traindir0 / f'Spv3.{n}.{m+1}{letter}.pt')
flow1.eval()


def dingo_mass_plot():
    from dtempest.gw.catalog import Catalog

    cat = 'gwtc-1'
    catalog = Catalog(cat)

    print(catalog.common_names)
    print(catalog.mergers)

    last = None
    bad_events = ['GW170817', 'GW151226', 'GW170608', 'GW151012', 'GW170809']
    colors = [color for color in TABLEAU_COLORS.keys() if color not in ['tab:gray']]

    gwtc = {f'{cat.upper()}: {event}': convert(CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5'))
            for event in catalog.common_names[:last] if event not in bad_events}

    def mass_only_convert(event):
        sdict = flow0.sample_dict(10000, context=Merger(event, cat).make_array())
        converted = convert({key: sdict[key] for key in ['chirp_mass', 'mass_ratio']})
        sdict.update({key: converted[key] for key in ['mass_1', 'mass_2']})
        return sdict

    sdicts = {f'{flow0.name}: {event}': mass_only_convert(event)
              for event in catalog.common_names[:last] if event not in bad_events}

    # gwtc.update(sdicts)

    multi_gwtc = CBCComparisonSampleDict(gwtc)
    multi_mine = CBCComparisonSampleDict(sdicts)
    del gwtc, sdicts

    select_params = ['mass_1', 'mass_2']

    kwargs = {
        'medians': None,  # f"Estimator {flow0.name}",
        'hist_bin_factor': 1,
        'bins': 20,
        'title_quantiles': [0.16, 0.5, 0.84],
        'smooth': 1.4,
        'label_kwargs': {'fontsize': 15},
        'labelpad': -0.1,
        'title_kwargs': {'fontsize': 15},
        'fill_contours': None,
        'plot_datapoints': False,
        'levels': (1 - np.exp(-9 / 2.),),
        'quantiles': None,

        'kde': stats.gaussian_kde,
        # 'kde': bounded_1d_kde,
        # 'kde_kwargs': multi.default_bounds(),

    }

    # The 3 sigma level
    fig = plt.figure(figsize=(8, 8))
    fig = multi_gwtc.plot(type='corner', parameters=select_params, colors=['tab:gray'] * len(multi_gwtc.labels),
                          fig=fig, **kwargs)
    fig.legends[0].remove()
    fig = multi_mine.plot(type='corner', parameters=select_params, colors=colors,
                          fig=fig, **kwargs)
    axs = fig.get_axes()
    for ax in axs:
        ax.set_xlim((10, 80))
    axs[2].set_ylim((10, 80))

    fig.savefig(f'{cat.upper()}_{flow0.name}_3sigma_mass.png', bbox_inches='tight')
    # plt.show()

    # The 1 sigma level
    kwargs['levels'] = (1 - np.exp(-0.5),)
    fig = plt.figure(figsize=(8, 8))
    fig = multi_gwtc.plot(type='corner', parameters=select_params, colors=['tab:gray'] * len(multi_gwtc.labels),
                          fig=fig, **kwargs)
    fig.legends[0].remove()
    fig = multi_mine.plot(type='corner', parameters=select_params, colors=list(TABLEAU_COLORS.keys()),
                          fig=fig, **kwargs)
    axs = fig.get_axes()
    for ax in axs:
        ax.set_xlim((10, 80))
    axs[2].set_ylim((10, 80))
    # plt.show()
    fig.savefig(f'{cat.upper()}_{flow0.name}_1sigma_mass.png', bbox_inches='tight')

    # The 90% level
    kwargs['levels'] = (0.9,)
    fig = plt.figure(figsize=(8, 8))
    fig = multi_gwtc.plot(type='corner', parameters=select_params, colors=['tab:gray'] * len(multi_gwtc.labels),
                          fig=fig, **kwargs)
    fig.legends[0].remove()
    fig = multi_mine.plot(type='corner', parameters=select_params, colors=list(TABLEAU_COLORS.keys()),
                          fig=fig, **kwargs)
    axs = fig.get_axes()
    for ax in axs:
        ax.set_xlim((10, 80))
    axs[2].set_ylim((10, 80))
    # plt.show()
    fig.savefig(f'{cat.upper()}_{flow0.name}_90%_mass.png', bbox_inches='tight')


def basic_corner():
    # testset = convert_dataset(catalog.mergers.values(), flow0.param_list)
    # sset0 = flow0.sample_set(3000, testset, name=f'flow {flow0.name}')

    cat = 'gwtc-1'
    # event = 'GW150914'
    event = 'GW170823'

    # cat = 'gwtc-2.1'
    # event = 'GW190814_211039' # .split('_')[0] # GW190814 is a bit special...

    # cat = 'gwtc-3'
    # event = 'GW200129_065458'
    # event = 'GW200208_222617'
    # event = 'GW200224_222234'
    # event = 'GW200220_061928'  # Not as good, as expected. Way outside learning prior
    # event = 'GW200308_173609'
    # gwtc = convert(SampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))

    merger = Merger(event, cat)

    if cat == 'gwtc-1':
        gwtc = convert(CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5'))
    elif cat == 'gwtc-2.1':
        gwtc = convert(CBCSampleDict.from_file(catalog_2 / f'{event}_cosmo.h5')['C01:Mixed'])
    elif cat == 'gwtc-3':
        gwtc = convert(CBCSampleDict.from_file(catalog_3 / f'{event}_cosmo.h5')['C01:Mixed'])
    else:
        gwtc = None

    # print(type(CBCSampleDict.from_samplesdict(gwtc)))
    # raise Exception

    # print(gwtc.parameters)

    # full = sset0.full_test()
    # full_rel = sset0.full_test(relative=True)

    # image = testset['images'][event]
    # label = testset['labels'][event]
    image = merger.make_array()
    sdict = flow0.sample_dict(20000, context=image)  # 20000 takes all my RAM for a few seconds...
    sdict1 = flow1.sample_dict(20000, context=image)

    multi = CBCComparisonSampleDict({cat.upper(): gwtc,
                                     f"{flow0.name}": sdict,
                                     f"{flow1.name} (Overfitted)": sdict1})

    del sdict, gwtc

    # fig = plt.figure(figsize=(12, 10))
    select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']

    # Sort-of-DONE: Rewrite pesummary comparison analysis. Open samples_dict, plot, configuration and corner.core
    # Sort-of-DONE: Pretty much done, but may need some extra things

    kwargs = {
        'medians': 'all',  # f"Estimator {flow0.name}",
        'hist_bin_factor': 1,
        'bins': 20,
        'title_quantiles': [0.16, 0.5, 0.84],
        'smooth': 1.4,
        'label_kwargs': {'fontsize': 20},
        # 'labelpad': 0.2,
        'title_kwargs': {'fontsize': 15},

        'kde': stats.gaussian_kde
        # 'kde': bounded_1d_kde,
        # 'kde_kwargs': multi.default_bounds(),
    }

    fig = multi.plot(type='corner', parameters=select_params, **kwargs)
    del multi
    plt.tight_layout(h_pad=-5, w_pad=-0.3)  # h_pad -1 for 1 line title, -3 for 2 lines
    # fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
    #                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
    fig = plot_image(image, fig=fig,
                     title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                     title_kwargs={'fontsize': 15})
    fig.get_axes()[-1].set_position(pos=[0.65, 0.65, 0.32, 0.32])

    from dtempest.core.common_utils import redraw_legend
    redraw_legend(fig, fontsize=15, loc='upper center', handlelength=2, linewidth=5)

    fig.savefig(f'{event}_{flow0.name}_{flow1.name}_comparison.png', bbox_inches='tight')
    # plt.show()


# dingo_mass_plot()
basic_corner()