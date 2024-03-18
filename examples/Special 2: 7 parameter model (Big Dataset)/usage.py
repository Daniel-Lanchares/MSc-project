from pathlib import Path
import matplotlib.pyplot as plt

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 2. 7 parameter model (Big Dataset)'
traindir0 = train_dir / 'training_test_6'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / 'Spv2.6.0.pt')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 999
event = f'{seed}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
dataset.change_parameter_name('d_L', to='luminosity_distance')
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')
sset0 = flow0.sample_set(3000, trainset[:][:], name=f'flow {flow0.name}')

full = sset0.full_test()
full_rel = sset0.full_test(relative=True)


#
# sdict = sset0[event]
# fig = sdict.plot(type='corner', truths=trainset['labels'][event])

image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(10000, context=image, reference=label)

fig = sdict.plot(type='corner', truths=label)  # TODO: check how to plot only certain parameters

fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])


# For discarding problematic samples
# ras, decs, dists = [], [], []
# for i, dec in enumerate(sdict['dec']):
#     if abs(dec) < 3.14/2:
#         ras.append(sdict['ra'][i])
#         decs.append(dec)
#         dists.append(sdict['d_L'][i])
# sdict['ra'] = np.array(ras)
# sdict['dec'] = np.array(decs)
# sdict['d_L'] = np.array(dists)

# fig = sdict.plot(type='skymap')


# print(both.xs('chirp_mass', level='parameters'))
print(full.pp_mean().to_markdown(tablefmt='github'))
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


''' May be starting to overfit. Discuss in memory. Higher dimensional models may fare better. Might reduce complexity.
Dataset 999

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