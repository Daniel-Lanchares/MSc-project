from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans


def load_40set_paths(seed1: int):
    # print('Loading path for combined Trainset ' + f'{seed1} to {seed1 + 39}')
    paths = [trainset_dir / '20_sets' / (f'{seed1} to {seed1 + 19}.' + ', '.join(params_list) + '.pt'),
             trainset_dir / '20_sets' / (f'{seed1 + 20} to {seed1 + 39}.' + ', '.join(params_list) + '.pt')]
    return paths


n = 14
m = 0
letter = ''
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
# model, weights, pre_process = get_extractor('resnet18')
#
# extractor_config = {
#     'n_features': 512,
#     'base_net': model(weights=weights)  # To allow different weights. Can be
# }

net_config = {
    'pytorch_net': True,
    'depths': [2, 2, 2, 2],
    'output_features': 128
}

pre_process = None

train_config = {
    'num_epochs': 2,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.0005,  # 0.00025,
    # 'weight_check_max_val': 1e8,
    # 'weight_check_max_iter': 40,
    'grad_clip': None,
    # 'sched_kwargs': {
    #     'type': 'StepLR',
    #     'step_size': 25,
    #     'gamma': 0.8,
    #     'verbose': True
    #     }
}

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'scales': {'chirp_mass': 80, 'luminosity_distance': 2000, 'theta_jn': 2 * np.pi, 'ra': 2 * np.pi, 'dec': np.pi},
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

# rng = np.random.default_rng(0)
# seeds = rng.choice(np.arange(0, 45), size=20, replace=False)
# seeds = list(range(40))
vali_seeds = 999
# print(seeds)
'''
20 sets
0: [33  9  7 44 14 38 24 39 17 23  5 28 26  0 42  2 22 19 29  1]

30 sets
[30 20 31 23 33  6 29 14 24 15 25  5 34 22 36 32 42 19 44 35  4 16  1 10
17 13 38  9  0 40]

40 sets
[35 43 23 40 33 41 12 38 34  2  0 10 39  6  7 19 11  1 37 44 18 30 20 14
 29 25 17  9 15 27 24 13 26 22 28 42  3  5  4 21]

'''
# paths = [trainset_dir / '20_sets' / (f'{20} to {39}.'+', '.join(params_list)+'.pt'),
#          trainset_dir / '20_sets' / (f'{40} to {59}.'+', '.join(params_list)+'.pt')]
# dataset = TrainSet.load(paths, name='20 to 60')

# dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
# dataset.change_parameter_name('d_L', to='luminosity_distance')

valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds))
valiset.change_parameter_name('d_L', to='luminosity_distance')

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
# dataset = convert_dataset(dataset, params_list)
# dataset.save(trainset_dir / '20_sets' / (f'{seeds[0]} to {seeds[-1]}.'+', '.join(params_list)+'.pt'))
valiset = convert_dataset(valiset, params_list)

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv1.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv1.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv1.{n}.{m}{letter}')
# print(flow.get_training_stage_seeds())

dataset_paths = [load_40set_paths(seed) for seed in [0, 40]]    # Add 80 once 100 to 120 works
flow.train(dataset_paths, traindir, [train_config, ] * 2, valiset, cutoff=40000)

# flow.train(dataset, traindir, train_config, valiset)
flow.save_to_file(traindir / f'{flow.name}.pt')

''' Loss log
2.14.0



2.10.0  Converges more slowly, but may prove better in the long run
Average train: 19.8844, Delta: -1.01347 (-4.84964%)
Average valid: 19.6006, Delta: -0.901136 (-4.39542%)

2.10.1  Went well up until loss ~14.65, so more complex might be better, if slightly.
Average train: 11.5332, Delta: -0.330475 (-2.7856%)
Average valid: 17.4595, Delta: 0.666455 (3.96864%)

2.10.1b  Went well up until epoch 4 (~14.43) , switching trainsets helped somewhat, may try to reload them dynamically.
Average train: 11.6418, Delta: -0.270511 (-2.27086%)
Average valid: 17.4354, Delta: 0.598754 (3.55625%)


2.9.0   Maybe going down in complexity doesn't actually help...
Average train: 15.2938, Delta: -0.135009 (-0.875045%)
Average valid: 15.6643, Delta: 0.138106 (0.889504%)


2.8.0   Might be a sign or better training or maybe just bad initialization
Average train: 50.6589, Delta: -1.70129 (-3.2492%)
Average valid: 50.5461, Delta: -1.53499 (-2.9473%)


2.7.0
Average train: 14.5297, Delta: -0.143281 (-0.976498%)
Average valid: 15.0941, Delta: -0.0676 (-0.445862%)


2.6.0   Even bigger dataset (40)
Average train: 15.3116, Delta: -0.228793 (-1.47225%)
Average valid: 15.5354, Delta: -0.0760148 (-0.486918%)

2.6.1
Average train: 14.2082, Delta: -0.154016 (-1.07237%)
Average valid: 16.0171, Delta: 0.111318 (0.69986%)


2.5.0
Average train: 14.3939, Delta: -0.0497773 (-0.34463%)
Average valid: 16.6333, Delta: -0.0911763 (-0.545168%)


2.4.0
Average train: 11.9844, Delta: 0.0708565 (0.594757%)
Average valid: 21.0958, Delta: -2.22238 (-9.5307%)


2.3.0   Making own Embedding net with torchvision.models.ResNet
Average train: 21.8809, Delta: -0.975154 (-4.26651%)
Average valid: 21.4363, Delta: -1.06902 (-4.75007%)


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

2.0.2   (Introduced embedding net overwriting search term in other log)

Average: 14.9293, Delta: -0.16511 (-1.09384%)

2.0.3
Average: 12.8223, Delta: -0.363184 (-2.75441%)

2.0.4
Average: 11.0735, Delta: -0.599326 (-5.13436%)

2.0.5
Average: 10.2084, Delta: -0.405751 (-3.82274%)
'''
