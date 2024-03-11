from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 1. 5 parameter model (Big Dataset)'
traindir = train_dir / 'training_test_0'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'luminosity_distance',
    'ra',
    'dec'
]
model, weights, pre_process = get_extractor('resnet18')

extractor_config = {
    'n_features': 512,
    'base_net': model(weights=weights)  # To allow different weights. Can be
}

train_config = {
    'num_epochs': 10,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 64,  # IDEA: Start with small, increase without changing datasets. Seems to work great
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.00025,  # 0.00025,
    'grad_clip': None,
    # 'sched_kwargs': {
    #     'type': 'StepLR',
    #     'step_size': 5,
    #     'gamma': 0.8,
    #     'verbose': True
    #     }
}

flow_config = {  # This config seems to top at log_prob ~14.5, tough it may slowly improve
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,  # Adding more seemed to help slightly

    'base_transform': trans.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 2,
        'num_transform_blocks': 3,
        'use_batch_norm': True
    },
    'middle_transform': trans.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': trans.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}

rng = np.random.default_rng(0)
seeds = rng.choice(np.arange(0, 45), size=20, replace=False)
print(seeds)
'''
0: [33  9  7 44 14 38 24 39 17 23  5 28 26  0 42  2 22 19 29  1]
1: [11 35 24  5 17  1 32 43 44  9 14 16 36 12 30]
2: [39 16 24 13 33 14  8 30 29 42 43  3 10 37 25]
3: [32 18 13 26 28 11  1 22  8  6  2 25 21  3  5]
4: [36 30 32  3 26 11 43 22 38 17 35 44 15 24 29]
5: [27 24 16 38  0 10  2 23 18 41 11 25 17  5 20]
6:
7:
'''

dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
dataset.change_parameter_name('d_L', to='luminosity_distance')

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
dataset = convert_dataset(dataset, params_list)

'''Flow creation'''
# flow = CBCEstimator(params_list, flow_config, extractor_config, name='Spv1.0.0',
#                     workdir=traindir, mode='extractor+flow', preprocess=pre_process)

'''Training continuation of previous model'''
flow = CBCEstimator.load_from_file(traindir / 'Spv1.0.5.pt')
flow.rename('Spv1.0.6')
for param in flow.model._embedding_net.parameters():
    param.requires_grad = True
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

flow.train(dataset, traindir, train_config)
flow.save_to_file(traindir / f'{flow.name}.pt')

# Idea Training the net after flow is settled was a good call, let's see how far can it be kept up!
''' Loss log

1.0.0
Average: 14.5687, Delta: -0.0819495 (-0.559356%)

1.0.1
Average: 14.2012, Delta: -0.0205076 (-0.144199%)

1.0.2
Average: 13.9419, Delta: -0.0102886 (-0.0737417%)

1.0.3
Average: 13.8662, Delta: -0.00347793 (-0.0250758%)

1.0.4
Average: 11.6813, Delta: -0.219709 (-1.84614%)

1.0.5
Average: 9.71031, Delta: -0.179198 (-1.812%)

1.0.6
Average: 8.34333, Delta: -0.267667 (-3.10843%)
'''