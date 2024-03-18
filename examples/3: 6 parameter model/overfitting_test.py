from pathlib import Path
import numpy as np

from dtempest.gw import CBCEstimator

from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3. 6 parameter model'
traindir = train_dir / 'overfitting_test'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'd_L',
    'ra',
    'dec'
]
model, weights, pre_process = get_extractor('resnet18')

extractor_config = {
    'n_features': 512,
    'base_net': model(weights=weights)  # To allow different weights. Can be
}

train_config = {
    'num_epochs': 30,
    'checkpoint_every_x_epochs': None,  # Not yet implemented
    'batch_size': 128,  # Almost no effect on RAM compared, but proportional to chance of nan loss at beginning
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,  # 0.00003,  # 0.00025,
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

seeds = 32
print(seeds)
dataset = load_rawsets(rawdat_dir, seeds2names(seeds))

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
dataset = convert_dataset(dataset, params_list)

'''Flow creation'''
flow = CBCEstimator(params_list, flow_config, extractor_config, name='overfitting_test',
                    workdir=traindir, mode='extractor+flow', preprocess=pre_process)

'''Training continuation of previous model'''
# flow = CBCEstimator.load_from_file(traindir / 'overfitting_test.pt')
# flow.rename('overfitting_test')

# from pprint import pprint
#
# pprint(flow.metadata['train_history'])

flow.train(dataset, traindir, train_config)
flow.save_to_file(traindir / f'{flow.name}.pt')

