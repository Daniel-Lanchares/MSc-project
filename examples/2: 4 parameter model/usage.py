from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

import torch
from torchvision import models
from CBC_estimator.core import Estimator

from CBC_estimator.core.net_utils import create_feature_extractor
from CBC_estimator.core.flow_utils import create_flow
from CBC_estimator.core.conversion_utils import convert_dataset, plot_images

'''
Right now this file is useless (outside of testing)
as the 4 parameter model has not yet been achieved
'''


files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '2: 4 parameter model'
traindir = train_dir / 'training_test_0'

params_list = [
    'chirp_mass',
    'chi_eff',
    'd_L',
    'NAP'
    ]

extractor_config = {
    'n_features': 1024,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }
flow_config = {  # As it is now, it explodes instantly
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,
    'base_transform_kwargs': {  # These I will study in more detail #TODO
        'hidden_dim': 4,
        'num_transform_blocks': 2,
        # 'num_bins': 8
        },
    }

dataset = []
for seed in range(1):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir/f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)#trainset_dir/'4_parameter_trainset.pt')
del dataset
pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons
processed_trainset = (pre_process(trainset[0]), trainset[1])
del trainset

# TODO: load_from_file
flow = Estimator(params_list, processed_trainset, traindir, flow_config, extractor_config, mode='extractor+flow')
flow.model.load_state_dict(torch.load(traindir / 'Model_state_dict.pt'))
flow.eval()


sdict = flow.sample_dict(10, processed_trainset[0][0])
pprint(sdict)