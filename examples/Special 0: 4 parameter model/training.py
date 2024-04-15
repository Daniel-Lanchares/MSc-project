from pathlib import Path
import numpy as np
import pandas as pd
from torchvision.models.resnet import Bottleneck

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 10
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / 'Special 0. 4 parameter model'
traindir = train_dir / f'training_test_{n}'

params_list = [
    'chirp_mass',
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
n_epochs = 20
train_config = {
    'num_epochs': n_epochs,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,  # 0.00025,
    'weight_check_max_val': 5e8,
    'weight_check_max_iter': 0,  # No weight check because it doesn't work on rq transforms
    'grad_clip': None,
    'sched_kwargs': {
        'type': 'cosine',
        'T_max': n_epochs,
        'verbose': True
        }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'input_dim': len(params_list),
    'context_dim': net_config['output_features'],
    'num_flow_steps': 5,

    'base_transform': trans.mask_piece_rq_autoreg,  # parameter reset doesn't work with quad transforms for some reason
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
    paths = [trainset_dir / '10_sets' / (f'{seed1 + offset} to {seed1 + offset + 9}.' + ', '.join(params_list) + '.pt')
             for offset in range(0, 40, 10)]
    return paths


# Model version within training test
m = 0
letter = 'b'

vali_seeds = 999
valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')
valiset = convert_dataset(valiset, params_list)

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv0.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv0.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv0.{n}.{m}{letter}')

# print(flow.model)

shuffle_rng = np.random.default_rng(seed=m)  # For reproducibility of 'random' shuffling of dataset
# dataset_paths = [load_40set_paths(seed) for seed in [0, 40, 80]]
# dataset_paths += [[trainset_dir / '10_sets' / (f'{120} to {129}.' + ', '.join(params_list) + '.pt'),
#                    trainset_dir / '10_sets' / (f'{130} to {139}.' + ', '.join(params_list) + '.pt')]]
paths = [trainset_dir / '10_sets' / (f'{0 + offset} to {0 + offset + 9}.' + ', '.join(params_list) + '.pt')
         for offset in range(0, 50, 10)]
dataset = TrainSet.load(paths, name='50k_test').sample(n=50000, random_state=shuffle_rng)

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
0.10.0b Resnet18 + big rq-autoreg. Took 4.7 hours
Average train: 23.9012, Delta: -1.43239 (-5.65411%)
Average valid: 24.0019, Delta: 0.0179278 (0.0747493%)

0.10.0 Resnet34 + big rq-autoreg. Took 8 hours and 50k images to get here, but (somewhat) more stable 
and "no overfitting on sight" (see usage section).
Average train: 23.0447, Delta: -0.106323 (-0.45926%)
Average valid: 22.9225, Delta: -0.150231 (-0.651117%)

0.9.0 Big rq-autoreg, not amazing but stable in training
Average train: 269.773, Delta: -16.9065 (-5.89736%)
Average valid: 267.891, Delta: -18.6445 (-6.5069%)

0.8.0b slightly bigger. Again, volatile
Average train: 42.7464, Delta: 2.35913 (5.84126%)
Average valid: 34.6077, Delta: -20.6478 (-37.3679%)

0.8.1b
Average train: 24.2573, Delta: -2.63413 (-9.79542%)
Average valid: 23.8086, Delta: -0.526204 (-2.16235%)

0.8.0 20k small rq_autoreg
Average train: 44.4778, Delta: -2.37086 (-5.06068%)
Average valid: 45.5355, Delta: -0.958494 (-2.06154%)

0.7.0 Using a 50k dataset for 3 epochs with normal training. 
Average train: 1643.15, Delta: -2959.34 (-64.2987%)
Average valid: 1143.78, Delta: -1315.09 (-53.4835%)

0.6.0 Using dingo's rq-coupling transform. Way too volatile
Average train: 108.498
Average valid: 21.2372

0.6.1
Average train: 19.1674
Average valid: 17.8642

0.5.0 Also huge. Instability might arise from saving and then loading
Average train: 13.831
Average valid: 13.686

0.5.1
Average train: 12.9721
Average valid: 12.9883

0.5.2 Forgot to change the rng seed from previous
Average train: 12.4907
Average valid: 12.6529

0.5.3 Started to overfit
Average train: 12.2123
Average valid: 12.4962

0.4.0 (only 80,000 images, huge model: consider freezing net) (Trains no more)
Average train: 22.0759
Average valid: 20.5045


0.3.0 (only 80,000 images)
Average train: 15.1832
Average valid: 14.0433


0.2.0
Average train: 14.2187
Average valid: 13.9473


0.1.0
Average train: 13.9747
Average valid: 13.8294

0.1.1 
Average train: 12.2655
Average valid: 12.647

0.1.1b (random shuffle) Train will probably be worse but closer validation (Pretty much what happened)
Average train: 12.3656
Average valid: 12.5558

0.1.2 Test of random sampling
Average train: 12.0698
Average valid: 12.3306

0.1.2b
Average train: 12.1205
Average valid: 12.4318


0.0.0
Average train: 12.9577
Average valid: 13.0934



5p model
1.4.2.B6 (first trainset, overfitted after)
Average train: 11.3257
Average valid: 12.0903          <---- Best!


'''
