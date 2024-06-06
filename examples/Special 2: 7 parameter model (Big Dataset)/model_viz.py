from pathlib import Path
import numpy as np
# import pandas as pd
# from torchvision.models.resnet import Bottleneck
from torchview import draw_graph

from dtempest.gw import CBCEstimator


import dtempest.core.flow_utils as trans

n = 13  # Training test number
m = 0  # Model version within training test
letter = 'b'
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

flow_config = {  # Smaller flow, hopefully doesn't overfit
    'scales': {'chirp_mass': 80, 'luminosity_distance': 2000, 'theta_jn': 2*np.pi, 'ra': 2*np.pi, 'dec': np.pi},
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

if m == 0:
    '''Flow creation'''
    flow = CBCEstimator(params_list, flow_config, net_config, name=f'Spv2.{n}.{m}{letter}',
                        workdir=traindir, mode='net+flow', preprocess=pre_process)
else:
    '''Training continuation of previous model'''
    flow = CBCEstimator.load_from_file(traindir / f'Spv2.{n}.{m - 1}{letter}.pt')
    flow.rename(f'Spv2.{n}.{m}{letter}')

model_graph2 = draw_graph(flow.model._embedding_net,
                          input_size=(1,3,128,128),
                          expand_nested=True,
                          graph_dir='RL',
                          roll=True,
                          depth=1,
                          save_graph=False,
                          filename='basic_embedding_level_1_RL')
model_graph2.visual_graph.render(format='pdf', directory='./graphs')

'''
To plot: pdf to png (350 DPI), edit in google slides, download slide as pdf and include graphics.
'''