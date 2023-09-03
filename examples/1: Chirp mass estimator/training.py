import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models
from CBC_estimator.core import Estimator

from CBC_estimator.core.conversion_utils import convert_dataset, plot_images
from CBC_estimator.core.train_utils import QTDataset, DataLoader
from CBC_estimator.core.conversion_utils import convert_dataset, plot_images
import CBC_estimator.core.flow_utils as trans

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '1: Chirp mass estimator'
traindir = train_dir / 'training_test_1'

params_list = [
    'chirp_mass'
]

extractor_config = {
    'n_features': 1024,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }
# flow_config = {  # As it is now, it explodes instantly
#     'input_dim': len(params_list),
#     'context_dim': extractor_config['n_features'],
#     'num_flow_steps': 5,
#     'base_transform_kwargs': {  # These I will study in more detail #TODO
#         'hidden_dim': 4,
#         'num_transform_blocks': 2,
#         # 'num_bins': 8
#         },
#     }
train_config = {
    'num_epochs': 20,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'SGD',  # 'Adam'
    'learning_rate': 0.005,  # Study how to change it mid-training
    'schel_kwargs': None
    }

flow_config = {  # This config seems to top at log_prob ~4, tough it may slowly improve
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,

    'base_transform': trans.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 4,
        'num_transform_blocks': 2,
        # 'num_bins': 8
    },
    'middle_transform': trans.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': trans.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}

# trainset = [np.array([]).reshape(0, 3, 128, 128),
#             np.array([]).reshape(0, 1)]
dataset = []
for seed in range(3):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir / f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)
del dataset

pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons


flow = Estimator(params_list, flow_config, extractor_config, train_config,
                 workdir=traindir, mode='extractor+flow', preprocess=pre_process)


flow.train(trainset, traindir, train_config)
flow.save_to_file(traindir/'test_new_save-load_format.pt')
