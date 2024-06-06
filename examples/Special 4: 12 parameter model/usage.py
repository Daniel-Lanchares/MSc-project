from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
n = 0
m = 0
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 4. 12 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv4.{n}.{m}{letter}.pt')
flow0.rename(f'Spv4.{n}.{m}{letter}')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 999
event = f'{seed}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
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

sset0 = flow0.sample_set(3000, trainset[:][:1], name=f'flow {flow0.name}')

full = sset0.full_test()
full_rel = sset0.full_test(relative=True)


#
# sdict = sset0[event]
# fig = sdict.plot(type='corner', truths=trainset['labels'][event])

image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(10000, context=image, reference=label)

smooth = 1.4
size = 10

mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
a = -1  # -0.8
b = -0.38

fig = plt.figure(figsize=(1.4*size, size))
select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']
fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
                 smooth=smooth, smooth1d=smooth, medians=True, fig=fig,
                 label_kwargs={'fontsize': 10}, labelpad=0.2, title_kwargs={'fontsize': 10})
plt.tight_layout(h_pad=a, w_pad=b)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

plt.show()


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

| parameters<br>(flow Spv4.0.0)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units   |
|---------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|---------|
| chirp_mass                      |   46.3122    |   46.8531   |             16.7971 |                           17.8948 |                            19.7237 | $\%$    |
| mass_ratio                      |    0.603975  |    0.613444 |             41.3898 |                           37.0575 |                            39.2189 | $\%$    |
| a_1                             |    0.51739   |    0.49569  |            309.789  |                           66.9127 |                            64.8023 | $\%$    |
| a_2                             |    0.506029  |    0.494565 |            344.158  |                           67.8552 |                            67.748  | $\%$    |
| tilt_1                          |    1.55296   |    1.5994   |             58.2108 |                           42.9595 |                            43.943  | $\%$    |
| tilt_2                          |    1.5898    |    1.53842  |             62.8501 |                           41.1852 |                            42.7039 | $\%$    |
| phi_jl                          |    3.18139   |    3.02592  |            338.358  |                           65.1258 |                            60.3694 | $\%$    |
| phi_12                          |    3.00308   |    3.12589  |            305.609  |                           68.1175 |                            64.3834 | $\%$    |
| luminosity_distance             | 1439.33      | 1502.02     |             44.9647 |                           38.7148 |                            50.3738 | $\%$    |
| theta_jn                        |    1.58639   |    1.54645  |             81.0592 |                           50.1615 |                            49.3198 | $\%$    |
| ra                              |    3.05609   |    3.17589  |            541.9    |                           51.6403 |                            64.2196 | $\%$    |
| dec                             |   -0.0255346 |    0.016705 |            200.707  |                          680.911  |                           732.503  | $\%$    |

tensor(4.9366, grad_fn=<NegBackward0>)
'''