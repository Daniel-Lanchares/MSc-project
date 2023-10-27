from pathlib import Path
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torchvision import models
from dtempest.core import Estimator

from dtempest.gw.conversion import convert_dataset, plot_images
from dtempest.gw.catalog import Catalog
import dtempest.core.flow_utils as trans

'''
Chirp mass estimation has been achieved but is yet far from ideal

Its decent on dataset 0 (used to train) and somewhat worse on
dataset 2 (not used to train), but can still be spot on sometimes

Need a more objective way of evaluating performance.
Regardless it is currently at a log_prob of 4, 4.5.
Hopefully can be reduced.
'''

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '1: Chirp mass estimator'
traindir0 = train_dir / 'training_test_0'
traindir1 = train_dir / 'training_test_1'

# extractor_config = {
#     'n_features': 1024,
#     'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# }
# flow_config = {  # This is probably a more flexible
#     'input_dim': len(params_list),
#     'context_dim': extractor_config['n_features'],
#     'num_flow_steps': 5,
#
#     'base_transform': trans.mask_affine_autoreg,
#     'base_transform_kwargs': {
#         'hidden_dim': 4,
#         'num_transform_blocks': 2,
#     },
#     'middle_transform': trans.random_perm_and_lulinear,
#     'middle_transform_kwargs': {
#
#     },
#     'final_transform': trans.random_perm_and_lulinear,
#     'final_transform_kwargs': {
#
#     }
# }

dataset = []
for seed in range(3, 4):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir / f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

#flow0 = Estimator.load_from_file(traindir1 / 'v0.1.0.pt')
flow1 = Estimator.load_from_file(traindir1 / 'v0.1.5.pt')
#flow0.eval()
flow1.eval()

trainset = convert_dataset(dataset, flow1.param_list, name='Dataset 3')

# Preprocessing can be done manually if preferred, passing preprocess = False to various flow methods
# trainset = flow.preprocess(trainset)

# n = '2.00005'  # '0.00017'
# sdict0 = flow0.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='MKI')
# sdict1 = flow1.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='v0.1.0')
#
# layout = np.array([['chirp_mass', ], ])
# fig = sdict0.plot_1d_hists(layout, style='bilby', label='shared')
# fig = sdict1.plot_1d_hists(layout, style='deserted', fig=fig, label='shared', same=True,
#                            title=r'$\bf{MKI}$ vs $\bf{v0.1.0}$')
#
# # Interestingly, the first model is more accurate but less precise as opposed to the second in n=4
# error0 = sdict0.accuracy_test(sqrt=True)
# error1 = sdict1.accuracy_test(sqrt=True)
# print(error0)
# print()
# print(error1)
# plt.show()

testset = convert_dataset(Catalog('gwtc-1').mergers.values(), ['chirp_mass'])
print(testset)

#sset0 = flow0.sample_set(50, trainset[:][:], name='v0.1.0')  # TODO: savefile / loadfile methods.
sset1 = flow1.sample_set(50, trainset[:][:], name='v0.1.5')  # They take some time to make
sset2 = flow1.sample_set(500, testset, name='gwtc-1')  # They take some time to make

print(sset2['GW170817'].accuracy_test(sqrt=True))  # MSE of ~45 Solar Masses. Will need to look into it
del sset2['GW170817']

#error0 = sset0.accuracy_test(sqrt=True)
error1 = sset1.accuracy_test(sqrt=True)
error2 = sset2.accuracy_test(sqrt=True)


# print(sset0[n])
#print(error0.mean(axis=1))
print()
print(error1.mean(axis=1))
print()
print(error2.mean(axis=1))  # Improves a bit. Will need to take a look at each individually
print()
# print(trainset['labels'][n])
# layout = np.array([['chirp_mass', ], ])
# fig = sset0[n].plot_1d_hists(layout, style='bilby', label='shared')
# fig = sset1[n].plot_1d_hists(layout, style='deserted', fig=fig, label='shared', same=True,
#                              title=r'$\bf{MKI}$ vs $\bf{v0.1.0}$')
# plt.show()
# Idea: Model family class: sharing common architecture. Example: Family 0.1: Same config as v0.1.0
'''
MSE improved, need to calculate average uncertainty range (precision_test) (v0.1.0 should be much better).
Interestingly, there seems to be little drop of accuracy between data the model was trained on and not (no overfitting)

The error is still to large, but there v0.1.0 might still be trained over
Training over helps but may require tons of epochs (may switch to test 2: larger dataset + Scheduler)

Variations at 5000 samples are small enough
|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.5509 | M_{O}     |

|            |   MSE from MKIbis | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           17.5734 | M_{O}     |

For Dataset 0 (Part of training)
|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.5572 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.1801 | M_{O}     |

For Dataset 2 (Not part of MKI training)

|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.9686 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |            12.239 | M_{O}     |

For Dataset 3 (Not part of neither training)

|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        18.4548 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0851 | M_{O}     |
'''


'''
|            |   MSE from v0.1.1 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.9916 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0809 | M_{O}     |


|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0974 | M_{O}     |

|            |   MSE from v0.1.5 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.3885 | M_{O}     |
'''