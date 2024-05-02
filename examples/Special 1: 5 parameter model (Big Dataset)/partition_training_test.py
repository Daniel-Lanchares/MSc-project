from pathlib import Path
import numpy as np
import pandas as pd
from torchvision.models.resnet import Bottleneck

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 5
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 1. 5 parameter model (Big Dataset)'
traindir = train_dir / f'training_test_{n}'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'luminosity_distance',
    'ra',
    'dec'
]

net_config = {
    'pytorch_net': True,
    'depths': [2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
    # 'block': Bottleneck,
    'output_features': 128
}

pre_process = None
n_epochs = 10
train_config = {
    'num_epochs': n_epochs,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,  # 0.00025,
    'weight_check_max_val': 1e5,
    'weight_check_max_iter': 80,
    'grad_clip': None,
    'sched_kwargs': {
        'type': 'cosine',
        'T_max': n_epochs,
        'verbose': True
        }
    # 'sched_kwargs': {
    #     'type': 'StepLR',
    #     'step_size': 25,
    #     'gamma': 0.8,
    #     'verbose': True
    #     }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'scales': {'chirp_mass': 80, 'luminosity_distance': 2000, 'theta_jn': 2*np.pi, 'ra': 2*np.pi, 'dec': np.pi},
    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 3,

    'base_transform': trans.d_rq_coupling_and_affine,
    'base_transform_kwargs': {
        'hidden_dim': 64,
        'num_transform_blocks': 5,
        'use_batch_norm': True
    },
    'middle_transform': trans.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': trans.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}


def load_40set_paths(seed1: int) -> pd.DataFrame | None:
    # print('Loading path for combined Trainset ' + f'{seed1} to {seed1 + 39}')
    paths = [trainset_dir / '20_sets' / (f'{seed1} to {seed1 + 19}.' + ', '.join(params_list) + '.pt'),
             trainset_dir / '20_sets' / (f'{seed1 + 20} to {seed1 + 39}.' + ', '.join(params_list) + '.pt')]
    return paths


# Model version within training test
m = 0
letter = ''

vali_seeds = 999
valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')
valiset = convert_dataset(valiset, params_list)

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv1.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv1.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv1.{n}.{m}{letter}')

# dataset_paths = [load_40set_paths(seed) for seed in [0, 40, 80]]
# flow.train(dataset_paths, traindir, [train_config, ] * 1, valiset, cutoff=40000)
# flow.save_to_file(traindir / f'{flow.name}.pt')
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

size = 60  # Images (thousands)
paths = [trainset_dir / '20_sets' / (f'{0 + offset} to {0 + offset + 19}.' + ', '.join(params_list) + '.pt')
         for offset in range(0, size, 20)]
shuffle_rng = np.random.default_rng(seed=m)
dataset = TrainSet.load(paths, name=f'{size}k_test').sample(n=size*1000, random_state=shuffle_rng)
flow.train(dataset,
           traindir,
           train_config,
           valiset, )
flow.save_to_file(traindir / f'{flow.name}.pt')

'''Loss Log
1.5.0 Technically 1.7.0. now with scaling
Average train: -4.71±0.372, Delta: -0.229 (5.11%)
Average valid: -3.0±0.873, Delta: -0.0377 (1.27%) <-- Actually a bit overfitted


1.6.0
Average train: 13.7067
Average valid: 13.2754

1.6.1
Average train: 12.9411
Average valid: 12.8607


1.5.1a Then started to overfit
Average train: 13.1281
Average valid: 12.9518


1.4.3 Random shuffle might work long term. Might not
Average train: 11.6143
Average valid: 12.1842

1.4.2.B6 (first trainset, overfitted after)
Average train: 11.3257
Average valid: 12.0903          <---- Best!

1.4.2.B5
Average train: 11.5635
Average valid: 12.2529

1.4.2.B4
Average train: 11.8179
Average valid: 12.2243

1.4.2.B3
Average train: 12.0702
Average valid: 12.2826

1.4.2.B1
Average train: 12.7645
Average valid: 12.7071

1.4.1.B0
Average train: 22.7151
Average valid: 15.8587

1.4.0.B1
Average train: 14.0132
Average valid: 13.2217

1.4.0.B2
Average train: 13.0418
Average valid: 12.9287

Next plan: 5-epoch (or 1) stages switching between trainsets. Need to generate more... Which turn out to be a nightmare
'''
