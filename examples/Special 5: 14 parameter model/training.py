import os
from pathlib import Path
import numpy as np
# import pandas as pd
# from torchvision.models.resnet import Bottleneck
from torchvision import transforms

from dtempest.gw import CBCEstimator

# from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names, load_rawsets_pool
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 6  # Training test number
m = 2  # Model version within training test
letter = 'LIGO-O2'
vali_seeds = range(990, 1000)  # 999
zero_pad = 4

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'mid range' / 'LIGO-O2'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 5. 14 parameter model'
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
    'phase',
    'psi'
]

net_config = {
    'pytorch_net': True,
    'depths': [2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
    # 'block': Bottleneck,
    'output_features': 128
}

pre_process = None #transforms.Compose([
        #transforms.Normalize((0, 0, 0), (1, 1, 1))])
n_epochs = 12
train_config = {
    'num_epochs': n_epochs,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,  # 64
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,  # 0.001,
    'weight_check_max_val': 5e2,
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
    'scales': {'chirp_mass': 60,  # 100 normal, 40 low, 120 high
               'tilt_1': np.pi,
               'tilt_2': np.pi,
               'phi_jl': 2*np.pi,
               'phi_12': 2*np.pi,
               'luminosity_distance': 5000,  # 5000 normal, 2000 low, 6000 high
               'theta_jn': 2 * np.pi,
               'ra': 2*np.pi,
               'dec': np.pi,
               'phase': 2*np.pi,
               'psi': np.pi},

    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 8,  # 16 seemed to train fine

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
#     paths = [trainset_dir / '10_sets' / (f'{seed1 + offset} to {seed1 + offset + 9}.'+', '.join(params_list) + '.pt')
#              for offset in range(0, 40, 10)]
#     return paths


valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds, zero_pad=zero_pad))
# valiset.change_parameter_name('d_L', to='luminosity_distance')
valiset = convert_dataset(valiset, params_list)

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv5.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv5.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv5.{n}.{m}{letter}')

# print(flow.model)

# shuffle_rng = np.random.default_rng(seed=m)  # For reproducibility of 'random' shuffling of dataset

seeds = range(240)
dataset = load_rawsets_pool(rawdat_dir, seeds2names(seeds, zero_pad=zero_pad), processes=os.cpu_count())

# dataset.change_parameter_name('d_L', to='luminosity_distance')
dataset = convert_dataset(dataset, params_list, name='MKIII 240k LIGO-O2')

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
5.0.0   ~3 hours, similar times to 10p model (10k less data, 2 extra epochs)
Average train: 8.26±0.441, Delta: -0.219 (-2.58%)
Average valid: 9.63±0.694, Delta: -0.0539 (-0.557%)

5.1.0   4.48 hours for 20 epochs @60k. Overfitted a bit though, from epoch 13 at ~-4.2 loss
Average train: -8.39±0.57, Delta: -0.124 (1.5%)
Average valid: -1.88±1.6, Delta: -0.0971 (5.45%)

5.2.0   3.5 hours for 12 epochs @60k. Normalizing images paid off. On the verge of overfitting perhaps.
Average train: -6.03±0.433, Delta: -0.195 (3.35%)
Average valid: -4.83±0.736, Delta: -0.0893 (1.88%)

5.3.0_dev ~12 min. Lightweight training to test library installation (5 epochs @10k)
Average train: -2.18±0.302, Delta: -0.57 (35.4%)
Average valid: -1.4±0.29, Delta: -0.27 (23.9%)

5.2.0_new_pipe_dev 2.5 hours for 12 epochs @60k. Good metrics but disappointing performance. Needs more epochs
    Better on sky position (except when it misses) and on long range dL, but worse on everything else.
Average train: -11.3±0.597, Delta: -0.36 (3.3%)
Average valid: -9.73±0.835, Delta: -0.0715 (0.74%)

5.2.0_new_pipe_2 5 hours for 18 epochs @60k.
Average train: -13.6±0.604, Delta: -0.17 (1.27%)
Average valid: -10.4±1.2, Delta: -0.00938 (0.0904%)

5.2.0MKIII 2 hours 45 mins for 12 epochs @60k. Decent sky position (though can miss) at the cost of chirp, q and a_i
Average train: -6.42±0.402, Delta: -0.256 (4.16%)
Average valid: -5.65±0.373, Delta: -0.0776 (1.39%)

5.2.0MKIIII_low_res ~ 1 hour 40 for 12 epochs @60k. Comparable training to higher res, a bit better performance.
Average train: -6.72±0.409, Delta: -0.314 (4.91%)
Average valid: -5.56±0.546, Delta: -0.075 (1.37%)

5.4.0   almost 4 hours for 10 epochs @120k low res
Average train: -7.28±0.459, Delta: -0.25 (3.55%)
Average valid: -6.24±0.714, Delta: -0.0185 (0.296%)

5.4.0b  4 and a half for 15 epochs at 100k, 10 step model
Average train: -7.83±0.44, Delta: -0.137 (1.78%)
Average valid: -5.86±0.641, Delta: 0.00154 (-0.0263%)

5.5.0 low chirp 4 hours for 12 epochs at 120k (8 step) Was starting to maybe overfit
Average train: -7.87±0.584, Delta: -0.198 (2.59%)
Average valid: -6.77±0.574, Delta: -0.0281 (0.417%)

5.5.0 high chirp 4 hours for 12 epochs at 120k (8 step) Was starting to maybe overfit
Average train: -7.71±0.414, Delta: -0.191 (2.54%)
Average valid: -6.28±0.636, Delta: -0.0322 (0.516%)

5.6.0 mid range, 5 hours 42 for 15 epochs at 180k (less res than before). Overfitted a bit.
Low spread on validation, that is promising. It is decent, but could be better.
Average train: -7.59±0.419, Delta: -0.108 (1.45%)
Average valid: -5.67±0.409, Delta: -0.0531 (0.946%)

5.6.0b mid range, 5 hours 7 minutes for 12 epochs at 200k. No apparent overfit
Average train: -7.09±0.418, Delta: -0.142 (2.05%)
Average valid: -6.21±0.425, Delta: -0.0339 (0.549%)

5.6.0LIGO-O2, 5 hours for 12 epochs at 200k (V1 = zero array)
Even less spread.
Average train: -6.51±0.372, Delta: -0.407 (6.67%)
Average valid: -5.19±0.247, Delta: -0.0697 (1.36%)

5.6.1LIGO-O2 6 hours 12 epochs at 240k (16 steps)  Very good, though might shallower may be just as good 
Average train: -8.09±0.372, Delta: -0.145 (1.82%)
Average valid: -6.15±0.338, Delta: -0.0297 (0.486%)
'''
