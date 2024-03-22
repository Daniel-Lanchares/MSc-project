from pathlib import Path
import numpy as np
import pandas as pd

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
    'depths': [2, 3, 4, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
    'output_features': 128
}

pre_process = None

train_config = {
    'num_epochs': 1,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.0001,  # 0.00025,
    'weight_check_max_val': 1e8,
    'weight_check_max_iter': 80,
    'grad_clip': None,
    # 'sched_kwargs': {
    #     'type': 'StepLR',
    #     'step_size': 25,
    #     'gamma': 0.8,
    #     'verbose': True
    #     }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 5,

    'base_transform': trans.mask_affine_autoreg,
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
    print('Loading path for combined Trainset ' + f'{seed1} to {seed1 + 39}')
    paths = [trainset_dir / '20_sets' / (f'{seed1} to {seed1 + 19}.' + ', '.join(params_list) + '.pt'),]
             # trainset_dir / '20_sets' / (f'{seed1 + 20} to {seed1 + 39}.' + ', '.join(params_list) + '.pt')]
    return paths

# from typing import get_args
# print(get_args(load_40set_paths.__annotations__['return']))
# if pd.DataFrame in get_args(load_40set_paths.__annotations__['return']):
#     print('True!!')
# else:
#     print('Nop')

# Model version within training test
m = 0

vali_seeds = 999
valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')
valiset = convert_dataset(valiset, params_list)

flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv1.{n}.{m}',
                    workdir=traindir, mode='net+flow', preprocess=pre_process)

dataset_paths = [load_40set_paths(seed) for seed in [0, 40, 80]]
flow.train(dataset_paths, traindir, [train_config,] * 2, valiset)
flow.save_to_file(traindir / f'{flow.name}.pt')
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])


'''Loss Log
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
