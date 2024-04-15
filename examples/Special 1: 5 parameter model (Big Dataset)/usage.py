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
train_dir = files_dir / 'Examples' / 'Special 1. 5 parameter model (Big Dataset)'
traindir0 = train_dir / 'training_test_4'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / 'Spv1.4.2.B6.pt')

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

Best model so far

| parameters<br>(flow Spv1.4.2.B6)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|------------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                         |   46.38      |   46.8531   |            6.17356  |                           7.14091 |                           7.34712  | $M_{\odot}$    |
| mass_ratio                         |    0.592085  |    0.613444 |            0.168598 |                           0.20229 |                           0.232095 | $ø$            |
| luminosity_distance                | 1470.63      | 1502.02     |          512.427    |                         552.524   |                         652.895    | $\mathrm{Mpc}$ |
| ra                                 |    3.05953   |    3.17589  |            1.11721  |                           1.30878 |                           1.2566   | $\mathrm{rad}$ |
| dec                                |   -0.0498375 |    0.016705 |            0.404739 |                           0.48882 |                           0.533604 | $\mathrm{rad}$ |

tensor(10.6815, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.4.2.B3)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|------------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                         |   45.6032   |   46.8531   |            6.62453  |                          7.69835  |                           7.84982  | $M_{\odot}$    |
| mass_ratio                         |    0.577473 |    0.613444 |            0.169911 |                          0.201764 |                           0.23304  | $ø$            |
| luminosity_distance                | 1396.15     | 1502.02     |          507.05     |                        546.574    |                         682.22     | $\mathrm{Mpc}$ |
| ra                                 |    2.83173  |    3.17589  |            1.20573  |                          1.38315  |                           1.39405  | $\mathrm{rad}$ |
| dec                                |    0.156133 |    0.016705 |            0.438065 |                          0.537821 |                           0.571709 | $\mathrm{rad}$ |

tensor(11.1456, grad_fn=<NegBackward0>)


| parameters<br>(flow Spv1.3.1b)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|----------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                       |   46.3576    |   46.8531   |             6.95897 |                          9.29351  |                           8.99562  | $M_{\odot}$    |
| mass_ratio                       |    0.612215  |    0.613444 |             0.17615 |                          0.225936 |                           0.212689 | $ø$            |
| luminosity_distance              | 1367.16      | 1502.02     |           536.337   |                        581.89     |                         693.561    | $\mathrm{Mpc}$ |
| ra                               |    3.11678   |    3.17589  |             1.21031 |                          1.41312  |                           1.43929  | $\mathrm{rad}$ |
| dec                              |    0.0674703 |    0.016705 |             0.44408 |                          0.616734 |                           0.593469 | $\mathrm{rad}$ |

tensor(11.6461, grad_fn=<NegBackward0>)

May be starting to overfit. Discuss in memory. Higher dimensional models may fare better. Might reduce complexity.

| parameters<br>(flow Spv1.0.0)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   48.4062   |   46.8531   |           12.1107   |                         14.272    |                          14.9117   | $M_{\odot}$    |
| mass_ratio                      |    0.589147 |    0.613444 |            0.19978  |                          0.250755 |                           0.245901 | $ø$            |
| luminosity_distance             | 1350.93     | 1502.02     |          631.218    |                        716.038    |                         811.363    | $\mathrm{Mpc}$ |
| ra                              |    3.01249  |    3.17589  |            1.5667   |                          1.83326  |                           1.89671  | $\mathrm{rad}$ |
| dec                             |    0.406344 |    0.016705 |            0.622412 |                          0.705924 |                           0.715433 | $\mathrm{rad}$ |

tensor(14.6704, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.0.1)   |       median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|--------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   43.0928    |   46.8531   |           12.1131   |                         12.9455   |                          13.3191   | $M_{\odot}$    |
| mass_ratio                      |    0.640272  |    0.613444 |            0.202112 |                          0.256885 |                           0.247457 | $ø$            |
| luminosity_distance             | 1482.14      | 1502.02     |          648.244    |                        712.264    |                         822.584    | $\mathrm{Mpc}$ |
| ra                              |    3.22662   |    3.17589  |            1.53758  |                          1.89336  |                           1.79857  | $\mathrm{rad}$ |
| dec                             |    0.0883979 |    0.016705 |            0.554149 |                          0.61339  |                           0.607522 | $\mathrm{rad}$ |

tensor(14.0581, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.0.2)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   44.4969   |   46.8531   |           11.7569   |                         12.5564   |                          12.752    | $M_{\odot}$    |
| mass_ratio                      |    0.570604 |    0.613444 |            0.202465 |                          0.238385 |                           0.235201 | $ø$            |
| luminosity_distance             | 1466.04     | 1502.02     |          649.183    |                        690.234    |                         808.174    | $\mathrm{Mpc}$ |
| ra                              |    3.21124  |    3.17589  |            1.53439  |                          1.90004  |                           1.76467  | $\mathrm{rad}$ |
| dec                             |    0.130841 |    0.016705 |            0.557726 |                          0.62378  |                           0.630918 | $\mathrm{rad}$ |

tensor(13.6882, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.0.4)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   45.7096   |   46.8531   |            8.03497  |                          6.16294  |                           5.72074  | $M_{\odot}$    |
| mass_ratio                      |    0.551911 |    0.613444 |            0.279271 |                          0.182682 |                           0.199462 | $ø$            |
| luminosity_distance             | 1547.1      | 1502.02     |          584.369    |                        429.896    |                         463.262    | $\mathrm{Mpc}$ |
| ra                              |    3.04146  |    3.17589  |            1.38326  |                          1.4028   |                           1.28899  | $\mathrm{rad}$ |
| dec                             |    0.231005 |    0.016705 |            0.79407  |                          0.420903 |                           0.438622 | $\mathrm{rad}$ |

tensor(10.7137, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.0.5)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   44.1255   |   46.8531   |            8.00341  |                          4.44973  |                           4.24537  | $M_{\odot}$    |
| mass_ratio                      |    0.52685  |    0.613444 |            0.198956 |                          0.135859 |                           0.146374 | $ø$            |
| luminosity_distance             | 1537.09     | 1502.02     |          594.134    |                        309.526    |                         323.092    | $\mathrm{Mpc}$ |
| ra                              |    2.61477  |    3.17589  |            1.41641  |                          0.893282 |                           0.837358 | $\mathrm{rad}$ |
| dec                             |    0.157072 |    0.016705 |            0.4943   |                          0.312752 |                           0.326509 | $\mathrm{rad}$ |

tensor(9.4282, grad_fn=<NegBackward0>)

| parameters<br>(flow Spv1.0.6)   |      median |       truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|---------------------------------|-------------|-------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                      |   46.0619   |   46.8531   |            7.76177  |                         3.5824    |                          3.47706   | $M_{\odot}$    |
| mass_ratio                      |    0.581955 |    0.613444 |            0.189059 |                         0.0981304 |                          0.0999786 | $ø$            |
| luminosity_distance             | 1531.3      | 1502.02     |          577.577    |                       250.235     |                        259.063     | $\mathrm{Mpc}$ |
| ra                              |    2.79729  |    3.17589  |            1.32719  |                         0.609008  |                          0.584719  | $\mathrm{rad}$ |
| dec                             |    0.110882 |    0.016705 |            0.478489 |                         0.245728  |                          0.25699   | $\mathrm{rad}$ |

tensor(7.8895, grad_fn=<NegBackward0>)
'''