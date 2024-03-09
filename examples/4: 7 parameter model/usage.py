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
train_dir = files_dir / 'Examples' / '4. 7 parameter model'
traindir0 = train_dir / 'training_test_0'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / 'v4.0.4.pt')
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

# fig = sdict.plot(type='corner', truths=trainset['labels'][event])  # TODO: check how to plot only certain parameters

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
print(full.pp_mean().loc[['chirp_mass']].to_markdown(tablefmt='github'))
# Idea: The model is incredible at estimating the average of a parameter over the entire dataset
# Idea: I suppose due to being trained with datasets with identical mass distribution (uniform 5 to 100 for each m)
# Idea: Might be interesting to make a dataset with different distributions
print()
# cross = full.xs('chirp_mass', level='parameters')
# print(cross.to_markdown(tablefmt='github'))
cross = full.loc[(slice(':'), ('chirp_mass',)), :]  # TODO: conversion_function.
print(cross.to_markdown(tablefmt='github'))

# for asymmetric precision
# print(precision[0].mean(axis=0))
# print()
# print(precision[1].mean(axis=0))
samples = flow0.sample_and_log_prob(3000, trainset['images'][0])
print(-torch.mean(samples[1]))
plt.show()


'''
Dataset 999

|                     |   MSE from v4.0.0 | units          |
|:--------------------|------------------:|:---------------|
| chirp_mass          |       2.43196e+09 | $M_{\odot}$    |
| mass_ratio          |       3.78908e+07 | $ø$            |
| chi_eff             |       3.51249e+07 | $ø$            |
| luminosity_distance |       2.40551e+09 | $\mathrm{Mpc}$ |
| theta_jn            |       2.91216e+07 | $\mathrm{rad}$ |
| ra                  |       2.15037e+07 | $\mathrm{rad}$ |
| dec                 |       5.51034e+07 | $\mathrm{rad}$ |

Log_prob 999.00001: 
tensor(30.9731, grad_fn=<NegBackward0>)



|                     |   MSE from v4.0.1 | units          |
|:--------------------|------------------:|:---------------|
| chirp_mass          |         12.503    | $M_{\odot}$    |
| mass_ratio          |          0.210396 | $ø$            |
| chi_eff             |          0.189867 | $ø$            |
| luminosity_distance |        638.538    | $\mathrm{Mpc}$ |
| theta_jn            |          0.652262 | $\mathrm{rad}$ |
| ra                  |          1.55502  | $\mathrm{rad}$ |
| dec                 |          0.548469 | $\mathrm{rad}$ |
tensor(15.7334, grad_fn=<NegBackward0>)



|                     |   MSE from v4.0.2 | units          |
|:--------------------|------------------:|:---------------|
| chirp_mass          |         12.2      | $M_{\odot}$    |
| mass_ratio          |          0.204297 | $ø$            |
| chi_eff             |          0.189478 | $ø$            |
| luminosity_distance |        644.287    | $\mathrm{Mpc}$ |
| theta_jn            |          0.636337 | $\mathrm{rad}$ |
| ra                  |          1.53791  | $\mathrm{rad}$ |
| dec                 |          0.548092 | $\mathrm{rad}$ |

tensor(15.2228, grad_fn=<NegBackward0>)


|                     |   MSE from v4.0.3 | units          |
|:--------------------|------------------:|:---------------|
| chirp_mass          |         12.1362   | $M_{\odot}$    |
| mass_ratio          |          0.215416 | $ø$            |
| chi_eff             |          0.209864 | $ø$            |
| luminosity_distance |        625.604    | $\mathrm{Mpc}$ |
| theta_jn            |          0.640329 | $\mathrm{rad}$ |
| ra                  |          1.53417  | $\mathrm{rad}$ |
| dec                 |          0.54861  | $\mathrm{rad}$ |
tensor(15.0655, grad_fn=<NegBackward0>)


|                     |   MSE from v4.0.4 | units          |
|:--------------------|------------------:|:---------------|
| chirp_mass          |         12.4004   | $M_{\odot}$    |
| mass_ratio          |          0.19666  | $ø$            |
| chi_eff             |          0.198233 | $ø$            |
| luminosity_distance |        644.169    | $\mathrm{Mpc}$ |
| theta_jn            |          0.636847 | $\mathrm{rad}$ |
| ra                  |          1.49294  | $\mathrm{rad}$ |
| dec                 |          0.551172 | $\mathrm{rad}$ |
tensor(15.0138, grad_fn=<NegBackward0>)

| parameters          |   accuracy |   precision_left |   precision_right | units          |
|:--------------------|-----------:|-----------------:|------------------:|:---------------|
| chirp_mass          |  12.6613   |        11.4386   |         12.046    | $M_{\odot}$    |
| mass_ratio          |   0.197298 |         0.218324 |          0.214036 | $ø$            |
| chi_eff             |   0.201247 |         0.251615 |          0.252261 | $ø$            |
| luminosity_distance | 649.62     |       555.453    |        716.008    | $\mathrm{Mpc}$ |
| theta_jn            |   0.639013 |         0.713762 |          0.731618 | $\mathrm{rad}$ |
| ra                  |   1.49301  |         1.69828  |          1.78792  | $\mathrm{rad}$ |
| dec                 |   0.549183 |         0.617645 |          0.618422 | $\mathrm{rad}$ |

| parameters          |   accuracy |   precision_left |   precision_right | units   |
|:--------------------|-----------:|-----------------:|------------------:|:--------|
| chirp_mass          |    36.6041 |          26.1216 |           27.7073 | $\%$    |
| mass_ratio          |    49.9252 |          37.7387 |           37.0357 | $\%$    |
| chi_eff             |   248.846  |         726.769  |          727.348  | $\%$    |
| luminosity_distance |    63.4224 |          41.173  |           53.5683 | $\%$    |
| theta_jn            |    81.6792 |          45.4862 |           46.6101 | $\%$    |
| ra                  |   586.361  |          56.3513 |           59.3324 | $\%$    |
| dec                 |   203.87   |        1165.07   |         1175.51   | $\%$    |
tensor(15.0948, grad_fn=<NegBackward0>)
'''