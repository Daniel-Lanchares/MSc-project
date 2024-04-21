from pathlib import Path
import numpy as np
# import pandas as pd
# from torchvision.models.resnet import Bottleneck

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 13  # Training test number
m = 0  # Model version within training test
letter = ''
vali_seeds = 999

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 2. 7 parameter model (Big Dataset)'
traindir = train_dir / f'training_test_{n}'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'luminosity_distance',
    'theta_jn',
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
n_epochs = 10  # 30
train_config = {
    'num_epochs': n_epochs,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,  # 0.001,
    # 'weight_check_max_val': 5e4,
    # 'weight_check_max_iter': 80,  # No weight check because it doesn't seem to need it
    'grad_clip': None,
    'sched_kwargs': {
        'type': 'cosine',
        'T_max': n_epochs,
        'verbose': True
        }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'scales': {'chirp_mass': 80, 'luminosity_distance': 1000, 'mass_ratio': 0.8},
    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 5,

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
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv2.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv2.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv2.{n}.{m}{letter}')

# print(flow.model)

shuffle_rng = np.random.default_rng(seed=m)  # For reproducibility of 'random' shuffling of dataset
# dataset_paths = [load_40set_paths(seed) for seed in [0, 40, 80]]
# dataset_paths += [[trainset_dir / '10_sets' / (f'{120} to {129}.' + ', '.join(params_list) + '.pt'),
#                    trainset_dir / '10_sets' / (f'{130} to {139}.' + ', '.join(params_list) + '.pt')]]
size = 40  # Images (thousands)
paths = [trainset_dir / '20_sets' / (f'{0 + offset} to {0 + offset + 19}.' + ', '.join(params_list) + '.pt')
         for offset in range(0, size, 20)]
dataset = TrainSet.load(paths, name=f'{size}k_test').sample(n=size*1000, random_state=shuffle_rng)

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
2.13.0  Changed scales: chp_m to 80, q to 0.8 (hopefully train better... Nop. better to downscale only)
Average train: 0.0528±0.38, Delta: -0.267 (-83.5%)  <-- First actual deviation data
Average valid: 1.43±0.87, Delta: 0.0125 (0.882%)

2.12.0e 4 and a half hours
Average train: -1.03±1.75, Delta: -0.123 (13.6%)
Average valid: 0.415±0.262, Delta: -0.0109 (-2.56%)


2.12.1d Might still have some juice in it. Looks worse on 999.1 than 2.12.0c. Could be statistics (seems to be).
Average train: -0.356±0.391, Delta: -0.288 (4.19e+02%)
Average valid: 0.367±0.477, Delta: -0.104 (-22.2%)

2.12.2d
Average train: -0.42±0.436, Delta: -0.0608 (16.9%)
Average valid: 0.264±0.694, Delta: -0.0167 (-5.95%)

2.12.0  0.75 hours for 1/3 the data and 1/3 the epochs, much faster and great loss. scaling payed off. Let's see hists.
Average train: 0.0584316, Delta: -0.320653 (-84.5861%)  <-- loss can take negative values. Investigate (negative det()?)
Average valid: 1.45495, Delta: 0.0183574 (1.27784%)

2.12.0b Double the dataset (40k) ~1.5 hours It's great. Almost too great. Lots of spread in validation though.
Average train: -0.128445, Delta: -0.220049 (-240.216%)
Average valid: 0.774911, Delta: -0.0561174 (-6.75276%)

2.12.0c Full 60k, 2.83 hours (SWAP memory clearly slower). Validation progress slowed but no apparent overfitting
Average train: -0.57144, Delta: -0.219326 (62.2882%)  <-- misleading % due to small numbers (~constant decrease)
Average valid: 0.554786, Delta: -0.0874426 (-13.6155%)

2.12.1c Started to overfit
Average train: -0.635002, Delta: -0.188783 (42.3072%)
Average valid: 0.441821, Delta: 0.0220383 (5.24994%)




2.11.0c Almost 9 hours. No progress on validation but by far best example of overfitted model. Try more params.
Average train: 3.73326, Delta: -0.168509 (-4.31879%)
Average valid: 46.1798, Delta: -0.26829 (-0.577612%)

2.11.0b mixing rq-coupling and affine looks promising. Minimal instabilities.
Average train: 16.4052, Delta: -0.25478 (-1.5293%)
Average valid: 16.5259, Delta: -0.293212 (-1.74332%)

2.11.1b Seems to be close to its limit, though not bad at all
Average train: 12.9182, Delta: -0.116433 (-0.893263%)
Average valid: 13.4413, Delta: -0.0827971 (-0.612219%)

2.11.2b 5 epochs and did pretty much nothing. That was all it wrote.
Average train: 12.8579, Delta: -0.0272555 (-0.211526%)
Average valid: 13.373, Delta: 0.00647292 (0.0484265%)

2.11.0 Using dingo's rq-coupling. Didn't seem to overfit at any point, but was probably about to. Less volatile. Yes it did
Average train: 24.0299, Delta: -0.884898 (-3.5517%)
Average valid: 23.9978, Delta: -0.424156 (-1.73678%)


2.10.0c
Average train: 28.0106, Delta: -2.03309 (-6.76711%)
Average valid: 26.9496, Delta: -15.5281 (-36.556%)  Had an instability on the previous. Not as meaningful an improvement


5p model
1.4.2.B6 (first trainset, overfitted after)
Average train: 11.3257
Average valid: 12.0903          <---- Best!


'''
