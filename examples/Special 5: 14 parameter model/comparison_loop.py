from pathlib import Path
import matplotlib.pyplot as plt
import json
from copy import copy

from dtempest.core.common_utils import redraw_legend

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict
from dtempest.gw.conversion import plot_image
from dtempest.gw.catalog import Catalog, full_names

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.gw.conversions import convert


import time
'''

'''
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


flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv5.{n}.{m}{letter}.pt')
flow0.eval()

estimation_dir = files_dir / 'Estimation Data' / flow0.name
corner_dir = estimation_dir / 'Corners'

# testset = convert_dataset(catalog.mergers.values(), flow0.param_list)
# sset0 = flow0.sample_set(3000, testset, name=f'flow {flow0.name}')

cat = 'gwtc-1'
# event = 'GW150914'
# event = 'GW170823'

# cat = 'gwtc-2.1'  # NANs in GW190720_000836????
# event = 'GW190814_211039' # .split('_')[0] # GW190814 is a bit special...

# cat = 'gwtc-3'
# event = 'GW200129_065458'
# event = 'GW200208_222617'
# event = 'GW200224_222234'
# event = 'GW200220_061928'  # Not as good, as expected. Way outside learning prior
# event = 'GW200308_173609'
# gwtc = convert(SampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))

# merger = Merger(event, cat)

plot_font_dict = {
    'labels': {
        'gwtc-1': 25, 'gwtc-2.1': 25, 'gwtc-3': 25
    },
    'medians': {
        'gwtc-1': 20, 'gwtc-2.1': 20, 'gwtc-3': 20
    },
    'title': {
        'gwtc-1': 40, 'gwtc-2.1': 40, 'gwtc-3': 40
    },
    'legend': {
        'gwtc-1': 25, 'gwtc-2.1': 30, 'gwtc-3': 30
    },
}

plt.style.use('draft.mplstyle')

merger_kwargs = {
    'img_res': (128, 128),  # (64, 96),
    'image_window': (-0.065, 0.075)#(-0.15, 0.1)
}
resol = merger_kwargs['img_res']

catalog = Catalog(cat, **merger_kwargs)
catalog.common_names.reverse()

print(catalog.common_names, len(catalog.common_names))

dtpst_data = dict()
gwtc_data = dict()
# with open(estimation_dir / f'{flow0.name}_{cat.upper()}.json', 'r') as fp:
#     dtpst_data = json.load(fp)
#
# with open(estimation_dir / f'{cat.upper()}.json', 'r') as fp:
#     gwtc_data = json.load(fp)

esp = 2

t1 = time.time()

for event in catalog.common_names:
    print('\n' * esp)
    print(f"Loading event '{event}'")
    print('\n' * esp)

    pycbc_name = copy(event)

    if cat == 'gwtc-1':
        if event == 'GW170817':
            load_kwargs = {'path_to_samples': 'IMRPhenomPv2NRT_highSpin_posterior'}
            gwtc = convert(
                CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5', **load_kwargs))
        # elif event == 'GW151012':
        #     continue
        else:
            gwtc = convert(CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5'))

    elif cat == 'gwtc-2.1':
        if event in [name for name in catalog.common_names if '_' not in name]:
            event = full_names[event]
            if pycbc_name == 'GW190425':
                gwtc = convert(
                    CBCSampleDict.from_file(catalog_2 / f'{event}_cosmo.h5')['C01:IMRPhenomPv2_NRTidal:HighSpin'])
            # simple_name = event.split('_')[0]
            else:
                gwtc = convert(CBCSampleDict.from_file(catalog_2 / f'{event}_cosmo.h5')['C01:Mixed'])
        else:
            gwtc = convert(CBCSampleDict.from_file(catalog_2 / f'{event}_cosmo.h5')['C01:Mixed'])

    elif cat == 'gwtc-3':
        gwtc = convert(CBCSampleDict.from_file(catalog_3 / f'{event}_cosmo.h5')['C01:Mixed'])
    else:
        gwtc = None

    image = catalog[pycbc_name].make_array()
    sdict = flow0.sample_dict(10000, context=image)  # 20000 takes all my RAM for a few seconds...

    multi = CBCComparisonSampleDict({cat.upper(): gwtc, f"Estimator {flow0.name}": sdict})

    del sdict, gwtc

    select_params = [param for param in flow0.param_list if param != 'geocent_time']
    #flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']

    event_data = multi.get_median_data(select_params, as_dict=True)
    dtpst_data[event] = event_data[f"Estimator {flow0.name}"]
    gwtc_data[event] = event_data[cat.upper()]

    with open(estimation_dir / f'{flow0.name}_{cat.upper()}.json', 'w') as fp:
        json.dump(dtpst_data, fp, indent=4)

    with open(estimation_dir / f'{cat.upper()}.json', 'w') as fp:
        json.dump(gwtc_data, fp, indent=4)

    print('\n' * esp)
    print(f"Event '{event}' loaded and ready to plot")
    print('\n' * esp)

    if event == 'GW170817':
        # Introduce some slight deviation to avoid nan limits
        multi[cat.upper()]['ra'][0] += 1e-5
        multi[cat.upper()]['ra'][-1] -= 1e-5

        multi[cat.upper()]['dec'][0] += 1e-5
        multi[cat.upper()]['dec'][-1] -= 1e-5

    kwargs = {
        'medians': 'all',  # f"Estimator {flow0.name}",
        'hist_bin_factor': 1,
        'bins': 20,
        'title_quantiles': [0.16, 0.5, 0.84],
        'smooth': 1.4,
        'label_kwargs': {'fontsize': plot_font_dict['labels'][cat]},  # 25 for GWTC-1, 25 for GWTC-2/3?
        'title_kwargs': {'fontsize': plot_font_dict['medians'][cat]},  # 20 for GWTC-1, 20 for GWTC-2/3?
        'kde': stats.gaussian_kde
    }

    fig = multi.plot(type='corner', parameters=select_params, **kwargs)
    del multi
    plt.tight_layout(h_pad=-4, w_pad=-0.8)  # h_pad -1 for 1 line title, -3 for 2 lines
    fig = plot_image(image, fig=fig,
                     title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                     title_kwargs={'fontsize': plot_font_dict['title'][cat]},  # 40 for GWTC-1, 40 for GWTC-2/3?
                     aspect=resol[1] / resol[0])
    fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

    redraw_legend(fig,
                  fontsize=plot_font_dict['legend'][cat],  # 25 for GWTC-1, 30 for GWTC-2/3
                  loc='upper center',
                  handlelength=2,
                  linewidth=5)

    fig.savefig(corner_dir / f'{event}_{flow0.name}_{cat.upper()}.pdf', bbox_inches='tight')

    print('\n' * esp)
    print(f"Event '{event}' saved")
    print('\n' * esp)

t2 = time.time()

dt = t2-t1
print(f'{t2-t1} seconds')
print(f'{(t2-t1)/60:.2f} minutes')


'''
Each analysis requires loading and converting reference samples (which is by far the bottleneck), sampling 10000 times 
and plotting the results, as well as saving median and 1 sigma deviation data.

Spv5.2.0 Analyses GWTC-1 in 7.52 minutes

Spv5.4.0 Analyses GWTC-1 in 7.70 minutes (11 events)
Spv5.4.0 Analyses GWTC-2.1 in 13.19 minutes (43 events, missing GW190720_000836 due to nan issues in V1 data)
Spv5.4.0 Analyses GWTC-3 in 10.89 minutes (35 events)

Spv5.5.0low_chirp Analyses GWTC-1 in 7.66 minutes (11 events)
Spv5.5.0low_chirp Analyses GWTC-2.1 in ? minutes (43 events, missing GW190720_000836 due to nan issues in V1 data)
Spv5.5.0low_chirp Analyses GWTC-3 in ? minutes (35 events)

Spv5.0.0V Analyses GWTC-1 in 7.69 minutes (11 events)
Spv5.0.0V Analyses GWTC-2.1 in 14.88 minutes (43 events)
Spv5.0.0V Analyses GWTC-3 in 12.03 minutes (35 events)
'''