from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
n = 6
m = 1
letter = 'LIGO-O2'
files_dir =  Path('/mnt/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'mid range' / 'GW170823'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 5. 14 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv5.{n}.{m}{letter}.pt')
flow0.rename(f'Spv5.{n}.{m}{letter}')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()
flow0.pprint_metadata()
raise Exception

zero_pad = 4

seed = 999
event = f'GW170823.{seed:0{zero_pad}}.00003'

dataset = load_rawsets(rawdat_dir, seeds2names(seed, zero_pad=zero_pad))
dataset.change_parameter_name('d_L', to='luminosity_distance')
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')

# flow0.scales = flow0._get_scales({'chirp_mass': 100})
# print(flow0.scales)
# # print(flow0.sample_and_log_prob(10, trainset['images'][event]))
# print(trainset.loc[:, 'labels'])
# print(flow0.rescale_trainset(trainset).loc[:, 'labels'])
#
# raise RuntimeError

# flow0.scales = flow0._get_scales({'chirp_mass': 100, 'luminosity_distance': 1000})

# sset0 = flow0.sample_set(3000, trainset[:][:1], name=f'flow {flow0.name}')
#
# full = sset0.full_test()
# full_rel = sset0.full_test(relative=True)


#
# sdict = sset0[event]
# fig = sdict.plot(type='corner', truths=trainset['labels'][event])

trigger = flow0.shifts[-1].item()
flow0.shifts[-1] = 0

image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(10000, context=image, reference=label)

from scipy import stats
kwargs = {
    'medians': 'all',  # f"Estimator {flow0.name}",
    'hist_bin_factor': 1,
    'bins': 20,
    'title_quantiles': [0.16, 0.5, 0.84],
    'smooth': 1.4,
    'label_kwargs': {'fontsize': 25},
    # 'labelpad': 0.2,
    'title_kwargs': {'fontsize': 20},

    'kde': stats.gaussian_kde,
    'hist_kwargs': {'density': True}
    # 'kde': bounded_1d_kde,
    # 'kde_kwargs': multi.default_bounds(),
}

select_params = flow0.param_list#[param for param in flow0.param_list if param != 'geocent_time']
truths = sdict.select_truths(select_params)
truths[-1] -= trigger

fig = sdict.plot(type='corner', parameters=select_params, truths=truths, **kwargs)
plt.tight_layout(h_pad=-3, w_pad=-0.3)  # h_pad -1 for 1 line title, -3 for 2 lines
# fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
#                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                 title_kwargs={'fontsize': 40})
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])
# fig.savefig(f'corner_{flow0.name}_{event}.png', bbox_inches='tight')
plt.show()

# smooth = 1.4
# size = 10
#
# mpl.rcParams['xtick.labelsize'] = 8
# mpl.rcParams['ytick.labelsize'] = 8
# a = -1  # -0.8
# b = -0.38
#
# fig = plt.figure(figsize=(1.4*size, size))
# select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']
# fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
#                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig,
#                  label_kwargs={'fontsize': 10}, labelpad=0.2, title_kwargs={'fontsize': 10})
# plt.tight_layout(h_pad=a, w_pad=b)
# fig = plot_image(image, fig=fig,
#                  title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
# fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

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
# cross = full.loc[(slice(':'), ('chirp_mass',)), :]
# print(cross.to_markdown(tablefmt='github'))

# for asymmetric precision
# print(precision[0].mean(axis=0))
# print()
# print(precision[1].mean(axis=0))
# samples = flow0.sample_and_log_prob(3000, trainset['images'][event])
# print(-torch.mean(samples[1]))
# plt.show()

'''
Dataset 999

| parameters<br>(flow Spv5.0.0)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   46.3917    |   46.8531   |            6.20444  |                          6.87024  |                           6.94874  | $M_{\odot}$    |
| mass_ratio                      |    0.597196  |    0.613444 |            0.167922 |                          0.195077 |                           0.226291 | $ø$            |
| a_1                             |    0.512993  |    0.49569  |            0.249843 |                          0.336487 |                           0.323629 | $ø$            |
| a_2                             |    0.497386  |    0.494565 |            0.248562 |                          0.327465 |                           0.324926 | $ø$            |
| tilt_1                          |    1.57571   |    1.5994   |            0.532152 |                          0.655419 |                           0.641718 | $\mathrm{rad}$ |
| tilt_2                          |    1.58815   |    1.53842  |            0.522843 |                          0.667404 |                           0.655013 | $\mathrm{rad}$ |
| phi_jl                          |    3.25771   |    3.02592  |            1.57272  |                          2.12291  |                           2.01809  | $\mathrm{rad}$ |
| phi_12                          |    3.09303   |    3.12589  |            1.56196  |                          2.09615  |                           2.08635  | $\mathrm{rad}$ |
| luminosity_distance             | 1465.03      | 1502.02     |          506.581    |                        550.911    |                         655.985    | $\mathrm{Mpc}$ |
| theta_jn                        |    1.48731   |    1.54645  |            0.630916 |                          0.761953 |                           0.845992 | $\mathrm{rad}$ |
| ra                              |    3.09939   |    3.17589  |            1.14349  |                          1.37117  |                           1.22041  | $\mathrm{rad}$ |
| dec                             |   -0.0519302 |    0.016705 |            0.427753 |                          0.503219 |                           0.565164 | $\mathrm{rad}$ |
| phase                           |    2.97795   |    3.20094  |            1.59451  |                          2.0562   |                           2.20711  | $\mathrm{rad}$ |
| psi                             |    1.63222   |    1.58768  |            0.776781 |                          1.08029  |                           1.03151  | $\mathrm{rad}$ |

tensor(5.8331, grad_fn=<NegBackward0>)


| parameters<br>(flow Spv4.0.0)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   46.3241    |   46.8531   |            6.45316  |                          7.53926  |                           7.702    | $M_{\odot}$    |
| mass_ratio                      |    0.604023  |    0.613444 |            0.168524 |                          0.208424 |                           0.211599 | $ø$            |
| a_1                             |    0.517124  |    0.49569  |            0.252356 |                          0.345071 |                           0.329448 | $ø$            |
| a_2                             |    0.505881  |    0.494565 |            0.250342 |                          0.341784 |                           0.336637 | $ø$            |
| tilt_1                          |    1.55258   |    1.5994   |            0.542985 |                          0.652695 |                           0.655276 | $\mathrm{rad}$ |
| tilt_2                          |    1.58886   |    1.53842  |            0.524997 |                          0.640236 |                           0.654499 | $\mathrm{rad}$ |
| phi_jl                          |    3.1799    |    3.02592  |            1.57168  |                          2.06547  |                           1.9139   | $\mathrm{rad}$ |
| phi_12                          |    3.0024    |    3.12589  |            1.5641   |                          2.04345  |                           1.92802  | $\mathrm{rad}$ |
| luminosity_distance             | 1439.7       | 1502.02     |          517.455    |                        572.434    |                         692.862    | $\mathrm{Mpc}$ |
| theta_jn                        |    1.58697   |    1.54645  |            0.628723 |                          0.797517 |                           0.778239 | $\mathrm{rad}$ |
| ra                              |    3.05394   |    3.17589  |            1.18028  |                          1.49955  |                           1.33557  | $\mathrm{rad}$ |
| dec                             |   -0.0242694 |    0.016705 |            0.433926 |                          0.52971  |                           0.577764 | $\mathrm{rad}$ |

'''