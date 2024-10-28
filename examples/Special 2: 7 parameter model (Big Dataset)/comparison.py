from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict

from dtempest.gw.conversion import plot_image
from dtempest.gw.catalog import Merger

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.gw.conversions import convert

'''

'''
n = 12
m = 0
letter = 'c'
files_dir = Path('/mnt/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 2. 7 parameter model (Big Dataset)'
traindir0 = train_dir / f'training_test_12'
traindir1 = train_dir / f'training_test_13'  # {n}
catalog_1 = files_dir / 'GWTC-1 Samples'
catalog_3 = files_dir / 'GWTC-3 Samples'

# flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv5.{n}.{m}{letter}.pt')
# flow0.eval()

flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv2.12.0c.pt')
flow0.rename('GP7')
flow0.eval()
flow0.pprint_metadata()
raise Exception
# flow1 = CBCEstimator.load_from_file(traindir1 / f'Spv2.13.0b.pt')
# flow1.eval()

# from pprint import pprint
# from copy import deepcopy

# meta0 = deepcopy(flow0.metadata)
# meta1 = deepcopy(flow1.metadata)
#
# del meta0['jargon'], meta1['jargon']

# print(flow0.name)
# pprint(meta0)
# print()
# print(flow1.name)
# pprint(meta1)

cat = 'gwtc-1'
# event = 'GW150914'
event = 'GW170823'

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

# catalog = Catalog(cat)

# seed = 999
# event = f'{seed}.00002'
#
# dataset = load_rawsets(rawdat_dir, seeds2names(seed))
# dataset.change_parameter_name('d_L', to='luminosity_distance')
# testset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')


# image = testset['images'][event]
# label = testset['labels'][event]
merger = Merger(event, cat, img_res=(128, 128), image_window=(-0.065, 0.075), old_pipe=True)
image = merger.make_array()
sdict0 = flow0.sample_dict(10000, context=image)
# sdict1 = flow1.sample_dict(10000, context=image)

multi = CBCComparisonSampleDict({"GWTC-1": gwtc,
                                 # "pycbc": convert(CBCSampleDict.from_file((
                                 #     "https://github.com/gwastro/2-ogc/raw/master/posterior_samples/"
                                 #     "H1L1V1-EXTRACT_POSTERIOR_150914_09H_50M_45UTC-0-1.hdf"),
                                 #     path_to_samples="samples")),
                                 f"Estimator {flow0.name}": sdict0,
                                 # f"Estimator {flow1.name}": sdict1,
                                 })
# multi = CBCComparisonSampleDict({f"Estimator {flow1.name}": sdict1, f"Estimator {flow0.name}": sdict0})

# fig = plt.figure(figsize=(12, 10))
select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']

# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
kwargs = {
    'medians': 'all',  # f"Estimator {flow0.name}",
    'hist_bin_factor': 2,
    'bins': 20,
    'title_quantiles': [0.16, 0.5, 0.84],
    'smooth': 1.4,
    'label_kwargs': {'fontsize': 20},
    # 'labelpad': 0.2,
    'title_kwargs': {'fontsize': 15},

    'kde': stats.gaussian_kde
    # 'kde': bounded_1d_kde,
    # 'kde_kwargs': sdict0.default_bounds(),
}
fig: plt.Figure = multi.plot(type='corner', parameters=select_params, **kwargs)
plt.tight_layout(h_pad=-3, w_pad=-0.3)  # h_pad -1 for 1 line title, -3 for 2 lines, -5 for 3 lines?
# fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
#                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                 title_kwargs={'fontsize': 20})
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

# from dtempest.core.common_utils import change_legend_loc
#
# change_legend_loc(fig, 'upper center')
from dtempest.core.common_utils import redraw_legend
redraw_legend(fig,
              fontsize=25,  # 25 for GWTC-1, 30 for GWTC-2/3
              loc='upper center',
              bbox_to_anchor=(0.4, 0.98),
              handlelength=2,
              linewidth=5)

# To remove gridlines
for ax in fig.get_axes():
    ax.grid(False)

# plt.savefig(f'{event}_{flow0.name}_vs_{flow1.name}.png', bbox_inches='tight')
# plt.savefig(f'{event}_{flow0.name}.png', bbox_inches='tight')
fig.savefig(f'{event}_{flow0.name}_corner.pdf', bbox_inches='tight')
# plt.show()
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


# print(both.xs('chirp_mass', level='parameters'))
# print(full.pp_mean().to_markdown(tablefmt='github'))
# Idea: The model is incredible at estimating the average of a parameter over the entire dataset
# Idea: I suppose due to being trained with datasets with identical mass distribution (uniform 5 to 100 for each m)
# Idea: Might be interesting to make a dataset with different distributions
# print()
# cross = full.xs('chirp_mass', level='parameters')
# print(cross.to_markdown(tablefmt='github'))
# cross = full.loc[(slice(':'), ('chirp_mass',)), :]  # TODO: conversion_function.
# print(cross.to_markdown(tablefmt='github'))

# for asymmetric precision
# print(precision[0].mean(axis=0))
# print()
# print(precision[1].mean(axis=0))
# samples, logprob = flow0.sample_and_log_prob(3000, trainset['images'][event])
# print(-torch.mean(logprob))


''' 
Dataset 999

10 epochs, 5 steps and extra scaling
| parameters<br>(flow Spv2.13.0b)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   47.0589    |   46.8531    |            6.09274  |                          7.21324  |                           7.24564  | $M_{\odot}$    |
| mass_ratio                        |    0.595772  |    0.613444  |            0.16696  |                          0.195558 |                           0.227806 | $ø$            |
| chi_eff                           |    0.0229094 |    0.0106512 |            0.15138  |                          0.189076 |                           0.185159 | $ø$            |
| luminosity_distance               | 1445.1       | 1502.02      |          508.218    |                        579.478    |                         697.912    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.57874   |    1.54645   |            0.629651 |                          0.790193 |                           0.807224 | $\mathrm{rad}$ |
| ra                                |    3.08003   |    3.17589   |            1.13373  |                          1.37282  |                           1.28497  | $\mathrm{rad}$ |
| dec                               |    0.0486615 |    0.016705  |            0.422808 |                          0.50007  |                           0.51038  | $\mathrm{rad}$ |

tensor(-7.9368, grad_fn=<NegBackward0>)



With 15 epochs on 8 flow steps
| parameters<br>(flow Spv2.12.0e)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.9626     |   46.8531    |            6.06843  |                          6.50902  |                           6.47826  | $M_{\odot}$    |
| mass_ratio                        |    0.59587    |    0.613444  |            0.165765 |                          0.186067 |                           0.214523 | $ø$            |
| chi_eff                           |    0.0245531  |    0.0106512 |            0.149719 |                          0.178095 |                           0.174689 | $ø$            |
| luminosity_distance               | 1473.97       | 1502.02      |          505.016    |                        557.215    |                         637.261    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.59081    |    1.54645   |            0.627239 |                          0.814805 |                           0.784108 | $\mathrm{rad}$ |
| ra                                |    3.10284    |    3.17589   |            1.10779  |                          1.32065  |                           1.16288  | $\mathrm{rad}$ |
| dec                               |    0.00524479 |    0.016705  |            0.409863 |                          0.47796  |                           0.513514 | $\mathrm{rad}$ |

tensor(-3.7287, grad_fn=<NegBackward0>)


Very promising. Should try with partition training, since bottleneck seems to be dataset size
| parameters<br>(flow Spv2.12.2d)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.7641     |   46.8531    |            6.20039  |                          7.29332  |                           7.40737  | $M_{\odot}$    |
| mass_ratio                        |    0.58499    |    0.613444  |            0.168269 |                          0.193812 |                           0.230616 | $ø$            |
| chi_eff                           |    0.0255881  |    0.0106512 |            0.151438 |                          0.19157  |                           0.184108 | $ø$            |
| luminosity_distance               | 1453.07       | 1502.02      |          506.387    |                        560.767    |                         663.704    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.52871    |    1.54645   |            0.628249 |                          0.754193 |                           0.835799 | $\mathrm{rad}$ |
| ra                                |    3.07529    |    3.17589   |            1.12716  |                          1.38749  |                           1.21794  | $\mathrm{rad}$ |
| dec                               |    0.00275306 |    0.016705  |            0.410235 |                          0.497041 |                           0.524447 | $\mathrm{rad}$ |

tensor(-2.9264, grad_fn=<NegBackward0>)


It is actually a lot better than this. There might be some outliers again
| parameters<br>(flow Spv2.12.0b)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.2602     |   46.8531    |            6.41877  |                          7.43916  |                           7.65078  | $M_{\odot}$    |
| mass_ratio                        |    0.59075    |    0.613444  |            0.168266 |                          0.199083 |                           0.230326 | $ø$            |
| chi_eff                           |    0.00966992 |    0.0106512 |            0.154607 |                          0.19325  |                           0.189644 | $ø$            |
| luminosity_distance               | 1398.58       | 1502.02      |          510.727    |                        571.823    |                         690.36     | $\mathrm{Mpc}$ |
| theta_jn                          |    1.55495    |    1.54645   |            0.630948 |                          0.770661 |                           0.818363 | $\mathrm{rad}$ |
| ra                                |    3.12848    |    3.17589   |            1.1736   |                          1.41299  |                           1.26849  | $\mathrm{rad}$ |
| dec                               |    0.00700715 |    0.016705  |            0.431095 |                          0.514045 |                           0.545946 | $\mathrm{rad}$ |

tensor(-2.2363, grad_fn=<NegBackward0>)


May be starting to overfit. Discuss in memory. Higher dimensional models may fare better. Might reduce complexity.

The outliers disappeared. Hurray!
| parameters<br>(flow Spv2.11.1b)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.6857    |   46.8531    |            6.40334  |                          7.46788  |                           7.56606  | $M_{\odot}$    |
| mass_ratio                        |    0.59372   |    0.613444  |            0.167201 |                          0.205522 |                           0.234237 | $ø$            |
| chi_eff                           |    0.0200059 |    0.0106512 |            0.153697 |                          0.196035 |                           0.18748  | $ø$            |
| luminosity_distance               | 1423.68      | 1502.02      |          504.259    |                        542.621    |                         643.14     | $\mathrm{Mpc}$ |
| theta_jn                          |    1.50703   |    1.54645   |            0.628357 |                          0.727088 |                           0.811886 | $\mathrm{rad}$ |
| ra                                |    2.97571   |    3.17589   |            1.2064   |                          1.35867  |                           1.39369  | $\mathrm{rad}$ |
| dec                               |    0.0330927 |    0.016705  |            0.443452 |                          0.556694 |                           0.562161 | $\mathrm{rad}$ |

tensor(10.2778, grad_fn=<NegBackward0>)


There must be some outliers there messing accuracies, but on the hole looks more promising

| parameters<br>(flow Spv2.11.0b)   |      median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|-------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   48.9862   |   46.8531    |            10.3157  |                         13.2431   |                          14.9606   | $M_{\odot}$    |
| mass_ratio                        |    0.688181 |    0.613444  |            11.3669  |                          0.561256 |                           0.649539 | $ø$            |
| chi_eff                           |    0.029012 |    0.0106512 |            36.3937  |                          0.311046 |                           0.310274 | $ø$            |
| luminosity_distance               | 1363.74     | 1502.02      |           610.314   |                        643.006    |                         936.173    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.61243  |    1.54645   |             0.85096 |                          0.852267 |                           0.875135 | $\mathrm{rad}$ |
| ra                                |    2.90556  |    3.17589   |             2.73766 |                          1.96147  |                           1.99715  | $\mathrm{rad}$ |
| dec                               |   -0.180777 |    0.016705  |            30.4319  |                          1.15556  |                           1.11038  | $\mathrm{rad}$ |

tensor(15.2436, grad_fn=<NegBackward0>)


rq-coupling isn't working either. Can't really understand why...

| parameters<br>(flow Spv2.11.0)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|----------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                       |   45.5633    |   46.8531    |           14.7641   |                          6.09637  |                           6.09549  | $M_{\odot}$    |
| mass_ratio                       |    0.529639  |    0.613444  |            0.218426 |                          0.519895 |                           0.523457 | $ø$            |
| chi_eff                          |    0.0198313 |    0.0106512 |            0.186828 |                          0.348805 |                           0.347934 | $ø$            |
| luminosity_distance              | 1233.25      | 1502.02      |          835.888    |                        358.505    |                         359.273    | $\mathrm{Mpc}$ |
| theta_jn                         |    1.44589   |    1.54645   |            0.630862 |                          0.70363  |                           0.703917 | $\mathrm{rad}$ |
| ra                               |    2.78381   |    3.17589   |            1.63494  |                          1.16108  |                           1.16174  | $\mathrm{rad}$ |
| dec                              |   -0.066339  |    0.016705  |            0.560934 |                          0.698657 |                           0.698254 | $\mathrm{rad}$ |

tensor(15.1434, grad_fn=<NegBackward0>)

Nope. rq-autoreg is definitely not working.

| parameters<br>(flow Spv2.10.0c)   |      median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|-------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   42.7519   |   46.8531    |           15.0181   |                          5.95563  |                           5.94535  | $M_{\odot}$    |
| mass_ratio                        |    0.39884  |    0.613444  |            0.272702 |                          0.447544 |                           0.44744  | $ø$            |
| chi_eff                           |   -0.063454 |    0.0106512 |            0.194349 |                          0.459363 |                           0.459419 | $ø$            |
| luminosity_distance               | 1131.05     | 1502.02      |          845.477    |                        293.727    |                         294.123    | $\mathrm{Mpc}$ |
| theta_jn                          |    1.31224  |    1.54645   |            0.648112 |                          0.71178  |                           0.712358 | $\mathrm{rad}$ |
| ra                                |    3.04301  |    3.17589   |            1.60138  |                          0.949061 |                           0.948814 | $\mathrm{rad}$ |
| dec                               |   -0.310683 |    0.016705  |            0.617699 |                          0.695898 |                           0.696072 | $\mathrm{rad}$ |

tensor(14.8169, grad_fn=<NegBackward0>)

Similar results than 2.0.3 at higher loss scores. Training better? And hopefully will start overfitting much latter.
| parameters<br>(flow Spv2.6.0)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   45.5484     |   46.8531    |            8.04104  |                         10.4546   |                          11.3747   | $M_{\odot}$    |
| mass_ratio                      |    0.658934   |    0.613444  |            0.187625 |                          0.266235 |                           0.244804 | $ø$            |
| chi_eff                         |    0.0361884  |    0.0106512 |            0.185129 |                          0.258259 |                           0.243397 | $ø$            |
| luminosity_distance             | 1224.4        | 1502.02      |          560.808    |                        548.528    |                         829.066    | $\mathrm{Mpc}$ |
| theta_jn                        |    1.5799     |    1.54645   |            0.644467 |                          0.814213 |                           0.802597 | $\mathrm{rad}$ |
| ra                              |    3.34396    |    3.17589   |            1.33388  |                          1.69941  |                           1.74131  | $\mathrm{rad}$ |
| dec                             |    0.00873722 |    0.016705  |            0.522891 |                          0.689662 |                           0.715275 | $\mathrm{rad}$ |

tensor(14.6519, grad_fn=<NegBackward0>)


| parameters<br>(flow Spv2.0.3)   |        median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|---------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   43.5927     |   46.8531    |            9.70473  |                         10.4539   |                           9.56819  | $M_{\odot}$    |
| mass_ratio                      |    0.632493   |    0.613444  |            0.204867 |                          0.241979 |                           0.208938 | $ø$            |
| chi_eff                         |    0.0728073  |    0.0106512 |            0.19148  |                          0.180189 |                           0.181267 | $ø$            |
| luminosity_distance             | 1772.41       | 1502.02      |          679.655    |                        659.548    |                         762.212    | $\mathrm{Mpc}$ |
| theta_jn                        |    1.53876    |    1.54645   |            0.637098 |                          0.773909 |                           0.793522 | $\mathrm{rad}$ |
| ra                              |    2.44652    |    3.17589   |            1.50347  |                          1.15096  |                           1.12332  | $\mathrm{rad}$ |
| dec                             |   -0.00374245 |    0.016705  |            0.506525 |                          0.535578 |                           0.627126 | $\mathrm{rad}$ |

tensor(12.2350, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv2.0.4)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   46.1104    |   46.8531    |            8.57825  |                          8.76803  |                           8.13167  | $M_{\odot}$    |
| mass_ratio                      |    0.600132  |    0.613444  |            0.195714 |                          0.225479 |                           0.192598 | $ø$            |
| chi_eff                         |    0.0279876 |    0.0106512 |            0.177515 |                          0.137855 |                           0.136697 | $ø$            |
| luminosity_distance             | 1149.21      | 1502.02      |          612.967    |                        351.973    |                         393.612    | $\mathrm{Mpc}$ |
| theta_jn                        |    1.42259   |    1.54645   |            0.639568 |                          0.725675 |                           0.74078  | $\mathrm{rad}$ |
| ra                              |    2.77478   |    3.17589   |            1.35026  |                          0.765093 |                           0.738342 | $\mathrm{rad}$ |
| dec                             |    0.216145  |    0.016705  |            0.536976 |                          0.476381 |                           0.547268 | $\mathrm{rad}$ |

tensor(9.9339, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv2.0.5)   |       median |        truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|--------------|--------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   44.9995    |   46.8531    |            8.84204  |                          7.73322  |                           7.19746  | $M_{\odot}$    |
| mass_ratio                      |    0.610448  |    0.613444  |            0.194441 |                          0.210886 |                           0.184904 | $ø$            |
| chi_eff                         |    0.0861766 |    0.0106512 |            0.199422 |                          0.13374  |                           0.133454 | $ø$            |
| luminosity_distance             | 1719.42      | 1502.02      |          618.671    |                        434.421    |                         461.96     | $\mathrm{Mpc}$ |
| theta_jn                        |    1.50243   |    1.54645   |            0.637958 |                          0.772239 |                           0.778773 | $\mathrm{rad}$ |
| ra                              |    3.35884   |    3.17589   |            1.30823  |                          0.632495 |                           0.62008  | $\mathrm{rad}$ |
| dec                             |   -0.043714  |    0.016705  |            0.498193 |                          0.398386 |                           0.451546 | $\mathrm{rad}$ |
'''
