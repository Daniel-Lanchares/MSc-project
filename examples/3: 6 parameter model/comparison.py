from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict

from dtempest.gw.conversion import plot_image
from dtempest.gw.catalog import Catalog

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.gw.conversions import convert

'''

'''
n = 5
m = 5
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3. 6 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_1 = files_dir / 'GWTC-1 Samples'
catalog_3 = files_dir / 'GWTC-3 Samples'

flow0 = CBCEstimator.load_from_file(traindir0 / f'v3.{n}.{m}{letter}.pt')
flow0.eval()


# testset = convert_dataset(catalog.mergers.values(), flow0.param_list)
# sset0 = flow0.sample_set(3000, testset, name=f'flow {flow0.name}')

cat = 'gwtc-1'
# event = 'GW150914'
event = 'GW150914'

# cat = 'gwtc-3'
# event = 'GW200129_065458'
# event = 'GW200224_222234'
# event = 'GW200220_061928'  # Not as good, as expected. Way outside learning prior
# gwtc = convert(SampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))

if cat == 'gwtc-1':
    gwtc = convert(CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5'))
elif cat == 'gwtc-3':
    gwtc = convert(CBCSampleDict.from_file(catalog_3 / f'{event}_cosmo.h5')['C01:Mixed'])
else:
    gwtc = None

# print(type(CBCSampleDict.from_samplesdict(gwtc)))
# raise Exception

# print(gwtc.parameters)

catalog = Catalog(cat)

# full = sset0.full_test()
# full_rel = sset0.full_test(relative=True)


# image = testset['images'][event]
# label = testset['labels'][event]
image = catalog[event].make_array()
sdict = flow0.sample_dict(5000, context=image)  # 20000 takes all my RAM for a few seconds...

multi = CBCComparisonSampleDict({"GWTC-1": gwtc, f"Estimator {flow0.name}": sdict})

del sdict, gwtc

# fig = plt.figure(figsize=(12, 10))
select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']

# Sort-of-DONE: Rewrite pesummary comparison analysis. Open samples_dict, plot, configuration and corner.core if needed
# Sort-of-DONE: Pretty much done, but may need some extra things

kwargs = {
    'medians': 'all',  # f"Estimator {flow0.name}",
    'hist_bin_factor': 1,
    'bins': 20,
    'title_quantiles': [0.16, 0.5, 0.84],
    'smooth': 1.4,
    'label_kwargs': {'fontsize': 15},
    'labelpad': -0.05,
    'title_kwargs': {'fontsize': 15},

    'kde': stats.gaussian_kde
    # 'kde': bounded_1d_kde,
    # 'kde_kwargs': multi.default_bounds(),
}

fig = plt.figure(figsize=(14, 10))
fig = multi.plot(type='corner', parameters=select_params, fig=fig, **kwargs)
del multi
plt.tight_layout(h_pad=-3, w_pad=-0.0)  # h_pad -1 for 1 line title, -3 for 2 lines
# fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
#                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                 title_kwargs={'fontsize': 15})
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

from dtempest.core.common_utils import change_legend_loc

change_legend_loc(fig, 'upper center')

# Problems with skymap as always. Cannot translate histogram to projection.
# import numpy as np
# corner_colors = ['#0072C1', '#b30909', '#8809b3', '#b37a09']
# corner_kwargs = dict(
#     bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
#     title_kwargs=dict(fontsize=16), color=corner_colors[0],
#     truth_color='tab:orange', quantiles=[0.16, 0.84],
#     levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
#     plot_density=False, plot_datapoints=True, fill_contours=True,
#     max_n_ticks=3
# )
#
# import corner
# import cartopy.crs as ccrs
# fig2 = plt.figure()
# ax = fig2.add_subplot(projection=ccrs.Mollweide())
# # ax.hist2d(sdict['ra'], sdict['dec'])
# corner.hist2d(sdict['ra'], sdict['dec'], ax=ax, bins=50, smooth=0.9,
#               color=corner_colors[0], levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.))
#               , plot_datapoints=True, fill_contours=True, new_fig=False)


# import numpy as np
# # For discarding problematic samples
# ras, decs, dists = [], [], []
# for i, dec in enumerate(sdict['dec']):
#     if abs(dec) < 3.14/2:
#         ras.append(sdict['ra'][i])
#         decs.append(dec)
#         dists.append(sdict['luminosity_distance'][i])
# sdict['ra'] = np.array(ras)
# sdict['dec'] = np.array(decs)
# sdict['luminosity_distance'] = np.array(dists)
#
# fig = sdict.plot(type='skymap', multi_resolution=False, distance_map=False)

fig.savefig(f'{event}_comparison_kde.png', bbox_inches='tight')
plt.show()
