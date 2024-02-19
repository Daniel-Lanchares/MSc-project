from pathlib import Path
import numpy as np

from dtempest.core import Estimator

from dtempest.core.common_utils import load_rawsets, seeds2names, get_extractor
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3. 6 parameter model'
traindir = train_dir / 'training_test_4'

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
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 256,  # Almost no effect on RAM compared, but proportional to chance of nan loss at beginning
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.00003,#0.00025,
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

rng = np.random.default_rng(4)
seeds = rng.choice(np.arange(0, 45), size=1, replace=False)
print(seeds)
'''
0: [ 9 20 38 43 10 32 26  1 21 16 44  2  0  6 40]
1: [11 35 24  5 17  1 32 43 44  9 14 16 36 12 30]
2: [39 16 24 13 33 14  8 30 29 42 43  3 10 37 25]
3: [32 18 13 26 28 11  1 22  8  6  2 25 21  3  5]
4: [36 30 32  3 26 11 43 22 38 17 35 44 15 24 29]
'''

dataset = load_rawsets(rawdat_dir, seeds2names(seeds))

# Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
dataset = convert_dataset(dataset, params_list)

'''Flow creation'''
# flow = Estimator(params_list, flow_config, extractor_config, name='v3.4.0',
#                  workdir=traindir, mode='extractor+flow', preprocess=pre_process)

'''Training continuation of previous model'''
flow = Estimator.load_from_file(traindir/'overfitting_test.pt')
flow.rename('overfitting_test')

# from pprint import pprint
# pprint(flow.metadata['train_history'])


flow.train(dataset, traindir, train_config)
flow.save_to_file(traindir/f'{flow.name}.pt')

'''
3.3.1
Average: 14.6242, Delta: -0.0119072 (-0.0813549%)

3.3.1 (Ups...)
Average: 14.6012, Delta: -0.0148934 (-0.101897%)

3.3.2
Average: ~14.49, Delta: ~-0.010 

3.3.3
Average: 14.4867, Delta: -0.010512 (-0.072511%)

3.3.4
Average: 14.382, Delta: -0.00916966 (-0.0637172%)

3.3.5
Average: 14.4043, Delta: -0.00924209 (-0.0641211%)
'''

'''
Deeper architecture

3.4.0
Average: 28.6662, Delta: -0.855203 (-2.89689%)

3.4.1
Average: 14.7765, Delta: -0.0769894 (-0.518324%)

3.4.2
Average: 14.3905, Delta: -0.0195544 (-0.135699%)

3.4.3
Average: 14.2216, Delta: -0.0153599 (-0.107887%)

test (only dataset 32, overfitting test to know whether the model can learn)
Average: 13.4706, Delta: -0.0815128 (-0.601477%)

overfitting_test
Average: 10.5968, Delta: -0.0199011 (-0.18745%)

Average: 10.2693, Delta: -0.020182 (-0.196143%)

Average: 9.88913, Delta: -0.0206944 (-0.208827%)

Average: 9.44537, Delta: -0.00135574 (-0.0143515%)

Average: 9.0541, Delta: -0.0134069 (-0.147857%)

Average: 8.73329, Delta: -0.00181274 (-0.0207524%)

Average: 8.4892, Delta: -0.00779676 (-0.0917591%)

Average: 8.2777, Delta: -0.00833778 (-0.100625%)

Average: 8.065, Delta: -0.00966291 (-0.11967%)

Average: 7.87069, Delta: -0.00107355 (-0.013638%)

Average: 7.70838, Delta: -0.00530605 (-0.0687875%)

Average: 7.57322, Delta: -0.00367107 (-0.0484509%)

Average: 7.44672, Delta: -0.00564899 (-0.0758013%)

Average: 7.3182, Delta: -0.00537462 (-0.073388%)

Average: 7.20049, Delta: -0.00406437 (-0.0564139%)

Average: 7.07996, Delta: -0.00460453 (-0.0649939%)

Average: 6.97038, Delta: -0.00519047 (-0.0744092%)
'''