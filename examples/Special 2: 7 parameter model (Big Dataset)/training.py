from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 2. 7 parameter model (Big Dataset)'
traindir = train_dir / 'training_test_1'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'luminosity_distance',
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
    'batch_size': 32,  # IDEA: Start with small, increase without changing datasets. Seems to work great
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

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 3,

    'base_transform': trans.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 2,
        'num_transform_blocks': 2,
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
vali_seeds = 999
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
valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
dataset.change_parameter_name('d_L', to='luminosity_distance')
valiset.change_parameter_name('d_L', to='luminosity_distance')

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
dataset = convert_dataset(dataset, params_list)
valiset = convert_dataset(valiset, params_list)

'''Flow creation'''
# flow = CBCEstimator(params_list, flow_config, extractor_config, name='Spv2.1.0',
#                     workdir=traindir, mode='extractor+flow', preprocess=pre_process)

'''Training continuation of previous model'''
flow = CBCEstimator.load_from_file(traindir / 'Spv2.1.1.pt')
flow.rename('Spv2.1.2')
for param in flow.model._embedding_net.parameters():
    param.requires_grad = True
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

flow.train(dataset, traindir, train_config, valiset)
flow.save_to_file(traindir / f'{flow.name}.pt')


''' Loss log

2.1.0
Average train: 16.7966, Delta: -0.367924 (-2.14351%) 
Average valid: 18.2569, Delta: -0.645987 (-3.4174%)

2.1.1
Average train: 14.8224, Delta: -0.207343 (-1.37955%)
Average valid: 15.8539, Delta: 0.0899254 (0.570451%)

2.1.2
Average train: 12.6756, Delta: -0.332469 (-2.55586%)
Average valid: 18.9634, Delta: 1.30666 (7.40034%)

2.0.0
Average: 19.1615, Delta: -0.376543 (-1.92723%)

2.0.1
Average: 16.599, Delta: -0.105823 (-0.633487%)

2.0.2 (Introduced embedding net ----- search term in other log)

Average: 14.9293, Delta: -0.16511 (-1.09384%)

2.0.3
Average: 12.8223, Delta: -0.363184 (-2.75441%)

2.0.4
Average: 11.0735, Delta: -0.599326 (-5.13436%)

2.0.5
Average: 10.2084, Delta: -0.405751 (-3.82274%)
'''