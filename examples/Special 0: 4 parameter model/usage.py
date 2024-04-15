from pathlib import Path
import matplotlib.pyplot as plt

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
n = 10
m = 0
letter = 'b'
seed = 999

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 0. 4 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / f'Spv0.{n}.{m}{letter}.pt')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()


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

fig = sdict.plot(type='corner', truths=trainset['labels'][event])  # TODO: check how to plot only certain parameters

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
samples = flow0.sample_and_log_prob(3000, trainset['images'].iloc[0])
print(-torch.mean(samples[1]))
plt.show()


''' 
Dataset 999

Just as bad if not worse than previous one

| parameters<br>(flow Spv0.10.0b)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-----------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                        |   46.6228   |   46.8531   |           14.7614   |                          5.33731  |                           5.33918  | $M_{\odot}$    |
| luminosity_distance               | 1610.67     | 1502.02     |          861.268    |                        356.533    |                         355.974    | $\mathrm{Mpc}$ |
| ra                                |    2.70367  |    3.17589  |            1.64861  |                          0.843748 |                           0.845232 | $\mathrm{rad}$ |
| dec                               |    0.221178 |    0.016705 |            0.570696 |                          0.60702  |                           0.606764 | $\mathrm{rad}$ |

tensor(12.4611, grad_fn=<NegBackward0>)

This one is heavily overtrained despite only reaching ~23 logprob on this dataset

| parameters<br>(flow Spv0.10.0)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|----------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                       |   46.9224   |   46.8531   |           14.7683   |                           5.61347 |                           5.63675  | $M_{\odot}$    |
| luminosity_distance              | 1640.81     | 1502.02     |          867.041    |                         382.644   |                         381.97     | $\mathrm{Mpc}$ |
| ra                               |    2.95021  |    3.17589  |            1.61174  |                           1.17482 |                           1.17567  | $\mathrm{rad}$ |
| dec                              |   -0.116434 |    0.016705 |            0.568479 |                           0.61702 |                           0.617128 | $\mathrm{rad}$ |

tensor(12.8481, grad_fn=<NegBackward0>)

Still somewhat underfitted, which is promising. Need to look up what training data I used

| parameters<br>(flow Spv0.8.1b)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|----------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                       |   22.1336   |   46.8531   |            25.7978  |                         19.2587   |                          18.4273   | $M_{\odot}$    |
| luminosity_distance              | 1002.6      | 1502.02     |           867.478   |                        687.507    |                         659.41     | $\mathrm{Mpc}$ |
| ra                               |    1.07282  |    3.17589  |             2.33795 |                          1.66736  |                           1.60089  | $\mathrm{rad}$ |
| dec                              |   -0.313841 |    0.016705 |             0.64376 |                          0.985884 |                           0.908364 | $\mathrm{rad}$ |

tensor(12.8217, grad_fn=<NegBackward0>)





Best model so far (5p model)

| parameters<br>(flow Spv1.4.2.B3)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|------------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                         |   45.6032   |   46.8531   |            6.62453  |                          7.69835  |                           7.84982  | $M_{\odot}$    |
| mass_ratio                         |    0.577473 |    0.613444 |            0.169911 |                          0.201764 |                           0.23304  | $ø$            |
| luminosity_distance                | 1396.15     | 1502.02     |          507.05     |                        546.574    |                         682.22     | $\mathrm{Mpc}$ |
| ra                                 |    2.83173  |    3.17589  |            1.20573  |                          1.38315  |                           1.39405  | $\mathrm{rad}$ |
| dec                                |    0.156133 |    0.016705 |            0.438065 |                          0.537821 |                           0.571709 | $\mathrm{rad}$ |

tensor(11.1456, grad_fn=<NegBackward0>)
'''