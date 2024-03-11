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
train_dir = files_dir / 'Examples' / '5. 8 parameter model'
traindir0 = train_dir / 'training_test_0'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / 'v5.0.1.pt')
flow0.change_parameter_name('d_L', to='luminosity_distance')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 999
event = f'{seed}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
dataset.change_parameter_name('d_L', to='luminosity_distance')
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')
sset0 = flow0.sample_set(3000, trainset[:][:10], name=f'flow {flow0.name}')

full = sset0.full_test()
full_rel = sset0.full_test(relative=True)


#
# sdict = sset0[event]
# fig = sdict.plot(type='corner', truths=trainset['labels'][event])

image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(3000, context=image, reference=label)

fig = sdict.plot(type='corner', truths=trainset['labels'][event])  # TODO: check how to plot only certain parameters

# fig = plot_image(image, fig=fig,
#                  title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
# fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])


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
samples = flow0.sample_and_log_prob(3000, trainset['images'][0])
print(-torch.mean(samples[1]))
plt.show()


'''
Dataset 999

| parameters<br>(flow v5.0.1)   |         median |         truth |   accuracy<br>(MSE) |   precision_left<br>(1.0$\sigma$) |   precision_right<br>(1.0$\sigma$) | units          |
|-------------------------------|----------------|---------------|---------------------|-----------------------------------|------------------------------------|----------------|
| chirp_mass                    |   46.2094      |   46.8531     |           12.2475   |                         11.7348   |                          12.2369   | $M_{\odot}$    |
| mass_ratio                    |    0.707557    |    0.613444   |            0.215918 |                          0.251935 |                           0.261881 | $ø$            |
| chi_1                         |    0.0471796   |   -0.00543866 |            0.254891 |                          0.342914 |                           0.343417 | unknown unit   |
| chi_2                         |   -0.0211244   |    0.0217712  |            0.246519 |                          0.316005 |                           0.319702 | unknown unit   |
| luminosity_distance           | 1297.65        | 1502.02       |          641.894    |                        601.082    |                         832.222    | $\mathrm{Mpc}$ |
| theta_jn                      |    1.59826     |    1.54645    |            0.634172 |                          0.799185 |                           0.8246   | $\mathrm{rad}$ |
| ra                            |    3.02992     |    3.17589    |            1.57102  |                          1.84225  |                           1.82095  | $\mathrm{rad}$ |
| dec                           |   -6.89269e-05 |    0.016705   |            0.553129 |                          0.644094 |                           0.635001 | $\mathrm{rad}$ |

tensor(15.4078, grad_fn=<NegBackward0>)
'''