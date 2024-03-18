from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

n = 3
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
# model, weights, pre_process = get_extractor('resnet18')
#
# extractor_config = {
#     'n_features': 512,
#     'base_net': model(weights=weights)  # To allow different weights. Can be
# }

net_config = {
    'pytorch_net': True,
    'depths': [2, 2, 2, 2],  # [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
    'output_features': 128
}

pre_process = None

train_config = {
    'num_epochs': 1,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.0001,  # 0.00025,
    'weight_check_max_val': 1e2,
    'weight_check_max_iter': 50,
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

# rng = np.random.default_rng(0)
# seeds = rng.choice(np.arange(0, 45), size=20, replace=False)
vali_seeds = 999
# print(seeds)

paths = [trainset_dir / '20_sets' / (f'{ 0} to {19}.'+', '.join(params_list)+'.pt'),
         trainset_dir / '20_sets' / (f'{20} to {39}.'+', '.join(params_list)+'.pt')]
dataset = TrainSet.load(paths, name='0 to 40')

# seeds = range(60, 80)
# dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
# dataset.change_parameter_name('d_L', to='luminosity_distance')

valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
# dataset = convert_dataset(dataset, params_list)
# dataset.save(trainset_dir / '20_sets' / (f'{seeds[0]} to {seeds[-1]}.'+', '.join(params_list)+'.pt'))
valiset = convert_dataset(valiset, params_list)

'''Flow creation'''
# flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv1.{n}.0',
#                     workdir=traindir, mode='net+flow', preprocess=pre_process)

'''Training continuation of previous model'''
flow = CBCEstimator.load_from_file(traindir / f'Spv1.{n}.1b.pt')
flow.rename(f'Spv1.{n}.1c')
# print(flow.get_training_stage_seeds())

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

flow.train(dataset, traindir, train_config, valiset)
flow.save_to_file(traindir / f'{flow.name}.pt')


'''Loss Log
1.1.0
Average train: 24.7031, Delta: -1.07871 (-4.18398%)
Average valid: 24.5754, Delta: -0.94382 (-3.69846%)

1.1.1 Peaked at epoch 4, Average valid: 13.5836, Delta: -0.07973 (-0.583533%). On the right track?
Average train: 11.5219, Delta: -0.317032 (-2.67789%)
Average valid: 15.8004, Delta: 0.873434 (5.8514%)


1.2.0 Doubled flow's hidden dims
Average train: 14.5718, Delta: -0.21357 (-1.44447%)
Average valid: 14.6372, Delta: -0.167498 (-1.13139%)

1.2.1 Peaked in epoch 3: Average valid: 13.1928, Delta: -0.0955684 (-0.719189%). Better than before, but nor by much...
Average train: 10.8299, Delta: -0.304528 (-2.73501%)
Average valid: 16.6025, Delta: 0.722667 (4.55086%)

1.3.0 Doubled again: Got an 83 loss start, we'll see if overfits on epoch 1...
Peaked on epoch 3: Average valid: 13.6245, Delta: -0.135512 (-0.984831%). 
Average train: 11.5876, Delta: -0.288085 (-2.42583%)
Average valid: 14.8296, Delta: 0.535765 (3.74823%)

1.3.1 Ideal for an overfitting test for TFM.
Average train: 9.80403, Delta: -0.41677 (-4.07766%)
Average valid: 18.675, Delta: 2.16363 (13.1039%)

1.3.1b Model at min logprob of 12.8493
Average train: 12.9557
Average valid: 12.8493

Next plan: 5-epoch (or 1) stages switching between trainsets. Need to generate more... Which turn out to be a nightmare
'''