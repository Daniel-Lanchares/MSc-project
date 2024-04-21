from pathlib import Path
import matplotlib.pyplot as plt

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
n = 0
m = 1
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 3. 10 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv3.{n}.{m}{letter}.pt')
flow0.rename(f'Spv3.{n}.{m}{letter}')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 32
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

fig = plt.figure(figsize=(12, 10))
select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']
fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
                 smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])


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
print(full_rel.pp_mean().to_markdown(tablefmt='github'))
# Idea: The model is incredible at estimating the average of a parameter over the entire dataset
# Idea: I suppose due to being trained with datasets with identical mass distribution (uniform 5 to 100 for each m)
# Idea: Might be interesting to make a dataset with different distributions
print()
# cross = full.xs('chirp_mass', level='parameters')
# print(cross.to_markdown(tablefmt='github'))
# cross = full.loc[(slice(':'), ('chirp_mass',)), :]  # TODO: conversion_function.
# print(cross.to_markdown(tablefmt='github'))

# for asymmetric precision
# print(precision[0].mean(axis=0))
# print()
# print(precision[1].mean(axis=0))
samples = flow0.sample_and_log_prob(3000, trainset['images'][event])
print(-torch.mean(samples[1]))
plt.show()

