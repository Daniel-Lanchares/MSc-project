from pathlib import Path
import numpy as np
# import pandas as pd
# from torchvision.models.resnet import Bottleneck

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 0  # Training test number
m = 0  # Model version within training test
letter = ''
vali_seeds = 999

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 4. 12 parameter model'
traindir = train_dir / f'training_test_{n}'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'a_1',
    'a_2',
    'tilt_1',
    'tilt_2',
    'phi_jl',
    'phi_12',
    'luminosity_distance',
    'theta_jn',
    'ra',
    'dec',
    # 'phase',
    # 'psi'
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
    'learning_rate': 0.001,  # 0.001,
    'weight_check_max_val': 3e1,
    'weight_check_tenfold': True,
    'weight_check_max_iter': 30,  # No weight check because it doesn't work on rq transforms
    'grad_clip': None,
    'sched_kwargs': {
        'type': 'cosine',
        'T_max': n_epochs,
        'verbose': True
        }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'scales': {'chirp_mass': 100, 'luminosity_distance': 1000},
    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 8,

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


# def load_40set_paths(seed1: int) -> pd.DataFrame | None:
#     # print('Loading path for combined Trainset ' + f'{seed1} to {seed1 + 39}')
#     paths = [trainset_dir / '10_sets' / (f'{seed1 + offset} to {seed1 + offset + 9}.' + ', '.join(params_list) + '.pt')
#              for offset in range(0, 40, 10)]
#     return paths


valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')
valiset = convert_dataset(valiset, params_list)

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv4.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv4.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv4.{n}.{m}{letter}')

# print(flow.model)

shuffle_rng = np.random.default_rng(seed=m)  # For reproducibility of 'random' shuffling of dataset

seeds = range(50)
dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
dataset.change_parameter_name('d_L', to='luminosity_distance')
dataset = convert_dataset(dataset, params_list)

# size = 50  # Images (thousands)
# paths = [trainset_dir / '10_sets' / (f'{0 + offset} to {0 + offset + 9}.' + ', '.join(params_list) + '.pt')
#          for offset in range(0, size, 10)]
# dataset = TrainSet.load(paths, name=f'{size}k_test').sample(n=size*1000, random_state=shuffle_rng)

# dataset = flow.preprocess(dataset)
#
# dataset = special_weight_check(flow, dataset,
#                                train_config['batch_size'],
#                                train_config['weight_check_max_val'],
#                                train_config['weight_check_max_iter'])

# flow.turn_net_grad('off')
flow.train(dataset,
           traindir,
           train_config,
           valiset, )
# cutoff=40000, trainset_random_state=shuffle_rng)
flow.save_to_file(traindir / f'{flow.name}.pt')
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])


'''Loss Log
4.0.0   ~2 hours, similar times to 10p model (10k less data)
Average train: 6.67±0.397, Delta: -0.298 (-4.28%)
Average valid: 7.31±0.625, Delta: -0.0965 (-1.3%)


'''
