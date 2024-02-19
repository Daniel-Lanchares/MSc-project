from pathlib import Path
import numpy as np

import torch
from torchvision import models
from dtempest.core import Estimator


from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as transf

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '2: 4 parameter model'
traindir = train_dir / 'training_test_3'

params_list = [
    'chirp_mass',
    'NAP',
    'chi_eff',
    'd_L',
]

extractor_config = {
    'n_features': 1024,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }

train_config = {
    'num_epochs': 100,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 128,
    'optim_type': 'SGD',  # 'Adam'
    'learning_rate': 0.0001,  # Study how to change it mid-training
    'grad_clip': 0.5,  # 1e-2 at the beginning,
    'sched_kwargs': {
        'type': 'StepLR',
        'step_size': 15,  # Note epochs go from 0 to num_epochs-1
        'gamma': 0.5,
        'verbose': True
        }
    }

flow_config = {  # This config seems to top at log_prob ~40, tough it may slowly improve
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,

    'base_transform': transf.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 4,
        'num_transform_blocks': 2,
        'use_batch_norm': True
    },
    'middle_transform': transf.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': transf.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}

dataset = []
for seed in range(8):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir / f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)
del dataset

pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons


# flow = Estimator(params_list, flow_config, extractor_config, train_config,
#                  workdir=traindir, mode='extractor+flow', preprocess=pre_process,
#                  name='v2.3.1_NAP')
flow = Estimator.load_from_file(traindir/'v2.3.3_NAP.pt')
flow.rename('v2.3.4_NAP')

print(flow.metadata['train_history']['stage 7']['training_time'])

flow.train(trainset, traindir, train_config)
flow.save_to_file(traindir/f'{flow.name}.pt')
