from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '5. 8 parameter model'
traindir = train_dir / 'training_test_0'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_1',
    'chi_2',
    'd_L',
    'theta_jn',
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
    'learning_rate': 0.001,  # 0.00025,
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
seeds = rng.choice(np.arange(0, 45), size=18, replace=False)
print(seeds)
'''
0: [18  8 15 35  6  0 44  2 41  1 40 27 23 30 20  9 31 24]
1: [11 35 24  5 17  1 32 43 44  9 14 16 36 12 30]
2: [39 16 24 13 33 14  8 30 29 42 43  3 10 37 25]
3: [32 18 13 26 28 11  1 22  8  6  2 25 21  3  5]
4: [36 30 32  3 26 11 43 22 38 17 35 44 15 24 29]
5: [27 24 16 38  0 10  2 23 18 41 11 25 17  5 20]
6:
7:
'''

dataset = load_rawsets(rawdat_dir, seeds2names(seeds))

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
dataset = convert_dataset(dataset, params_list)

'''Flow creation'''
# flow = CBCEstimator(params_list, flow_config, extractor_config, name='v5.0.0',
#                     workdir=traindir, mode='extractor+flow', preprocess=pre_process)

'''Training continuation of previous model'''
flow = CBCEstimator.load_from_file(traindir / 'v5.0.0.pt')
flow.rename('v5.0.1')
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

flow.train(dataset, traindir, train_config)
flow.save_to_file(traindir / f'{flow.name}.pt')

''' Loss log

5.0.0
Average: 16.491, Delta: -0.0463191 (-0.280088%)

5.0.1
Average: 16.0481, Delta: -0.0277329 (-0.172514%)
'''