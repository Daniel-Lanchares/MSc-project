from pathlib import Path

from torchvision import models
from dtempest.core import Estimator

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3: 6 parameter model'
traindir = train_dir / 'training_test_1'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'd_L',
    'ra',
    'dec'
]

extractor_config = {
    'n_features': 512,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }

train_config = {
    'num_epochs': 10,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 128,
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.0005,
    'grad_clip': None,
    # 'sched_kwargs': {
    #     'type': 'StepLR',
    #     'step_size': 15,  # Note epochs go from 0 to num_epochs-1
    #     'gamma': 0.5,
    #     'verbose': True
    #     }
    }

flow_config = {  # This config seems to top at log_prob ~4, tough it may slowly improve
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 3,  # Adding more seemed to help slightly

    'base_transform': trans.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 2,
        'num_transform_blocks': 2,
        'use_batch_norm': True
    },
    'middle_transform': trans.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': trans.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}

dataset = load_rawsets(rawdat_dir, seeds2names(range(15)))

trainset = convert_dataset(dataset, params_list)

del dataset

pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons


# flow = Estimator(params_list, flow_config, extractor_config, name='v3.1.0',
#                  workdir=traindir, mode='extractor+flow', preprocess=pre_process)
flow = Estimator.load_from_file(traindir/'v3.1.0.pt')
# from pprint import pprint
# pprint(flow.metadata['train_history'])
flow.rename('v3.1.1')

flow.train(trainset, traindir, train_config)
flow.save_to_file(traindir/f'{flow.name}.pt')
