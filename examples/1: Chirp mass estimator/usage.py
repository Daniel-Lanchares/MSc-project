from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision import models
from CBC_estimator.core import Estimator

from CBC_estimator.core.conversion_utils import convert_dataset, plot_images
import CBC_estimator.core.flow_utils as trans

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

params_list = ['chirp_mass']

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
#     'base_transform_kwargs': {  # These I will study in more detail #TODO
#         'hidden_dim': 4,
#         'num_transform_blocks': 2,
#         # 'num_bins': 8
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
for seed in range(2, 3):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir / f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)
del dataset


# pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons
# processed_trainset = (pre_process(trainset[0]), trainset[1])
# del trainset
#
# flow = Estimator(params_list, flow_config, extractor_config,
#                  train_config=None, workdir=traindir, mode='extractor+flow',
#                  preprocess=pre_process)
# flow.model.load_state_dict(torch.load(traindir / 'Model_state_dict.pt'))
#
# # This save/load ordeal has gone almost too smoothly... Need to test with custom transform functions though
#
# flow.save_to_file('MK I.pt')

flow0 = Estimator.load_from_file(traindir0 / 'MK I.pt')
flow1 = Estimator.load_from_file(traindir1 / 'test_new_save-load_format.pt')
flow0.eval()
flow1.eval()

# Preprocessing can be done manually if preferred, passing preprocess = False to various flow methods
# trainset = flow.preprocess(trainset)

n = 4  # 16
sdict0 = flow0.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='sdict1')
sdict1 = flow1.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='sdict2')

layout = np.array([['chirp_mass', ], ])
fig = sdict0.plot_1d_hists(layout, style='bilby', label='shared')
fig = sdict1.plot_1d_hists(layout, style='deserted', fig=fig, label='shared', same=True)

# Interestingly, the first model is more accurate but less precise as opposed to the second in n=4
print(sdict0.accuracy_test(total=False))
print(sdict1.accuracy_test(total=False))
plt.show()
