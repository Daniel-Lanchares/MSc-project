import os
from pathlib import Path
import numpy as np
# import pandas as pd
# from torchvision.models.resnet import Bottleneck
# from torchvision import transforms

from dtempest.gw import CBCEstimator

# from dtempest.core.train_utils import TrainSet
from dtempest.core.common_utils import load_rawsets, seeds2names, load_rawsets_pool
from dtempest.gw.conversion import convert_dataset
import dtempest.core.flow_utils as trans

if __name__ == '__main__':
    name = 'PlaceHolder' # Model name
    new_model = True  # Train from scratch (see later)
    dataset_name = 'example set'
    seeds = range(5)  # The seed range generated
    vali_seeds = 900  # Same
    zero_pad = 3


    files_dir = Path('')
    rawdat_dir = files_dir / 'Raw Datasets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'training_test_1' # This one is created automatically, the rest are not

    # If you are always going to train the same parameters, converting rawsets to trainsets may reduce loading times
    trainset_dir = files_dir / 'Trainsets'

    params_list = [
        'chirp_mass',
        'mass_ratio',
        'a_1',
        'a_2',
        'tilt_1',
        'tilt_2',
        'phi_jl',
        'phi_12',
        'luminosity_distance',
        'theta_jn',
        'ra',
        'dec',
        'phase',
        'psi',
        # 'normalized_time'
        # 'geocent_time'  # If we can get implement what was discussed about priors it could be a 15p estimation, but
        # don't think it would work right now
    ]

    net_config = {
        'pytorch_net': True,
        'depths': [2, 2, 2, 2],  # [2, 2, 2, 2] for 18, [3, 4, 6, 3] for resnet 34 and with BottleNeck for resnet50
        # 'block': Bottleneck,
        'output_features': 128
    }

    # I disabled image preprocessing for lack of RAM space, but in a cluster would probably be a good thing to do.
    # pre_process = transforms.Compose([
    #         transforms.Normalize((0, 0, 0), (1, 1, 1))])
    pre_process = None

    n_epochs = 5  # Test, probably will need more
    train_config = {
        'num_epochs': n_epochs,
        'checkpoint_every_x_epochs': None,  # Not yet implemented
        'batch_size': 64,  # 64
        'optim_type': 'Adam',  # 'SGD'
        'learning_rate': 0.001,  # 0.001,

        # With scaling redrawing weights is no longer relevant, for what I trained at least. Same with gradient clipping
        # 'weight_check_max_val': 1e2,
        # 'weight_check_tenfold': True,
        # 'weight_check_max_iter': 30,  # No weight check because it doesn't work on rq transforms
        'grad_clip': None,
        'sched_kwargs': {
            'type': 'cosine',
            'T_max': n_epochs,
            'verbose': True
            }
    }

    flow_config = {
        'scales': {'chirp_mass': 100.0,  # 100 normal, 40 low, 120 high
                   'tilt_1': np.pi,
                   'tilt_2': np.pi,
                   'phi_jl': 2*np.pi,
                   'phi_12': 2*np.pi,
                   'luminosity_distance': 5000.0,  # 5000 normal, 2000 low, 6000 high
                   'theta_jn': 2 * np.pi,
                   'ra': 2*np.pi,
                   'dec': np.pi,
                   'phase': 2*np.pi,
                   'psi': np.pi,
                   'geocent_time': 24*3600.0},
        # 'shifts': {'geocent_time': 1187529256.5},  # For tc testing, not fully implemented

        'input_dim': len(params_list),
        'context_dim': net_config['output_features'],

        # Here we can tweak the flow architecture. This is the structure of GP14 and LH14
        'num_flow_steps': 8,  # 16 also seemed to train fine
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



    valiset = load_rawsets(rawdat_dir, seeds2names(vali_seeds, zero_pad=zero_pad))
    valiset = convert_dataset(valiset, params_list)
    # valiset.save(filename)  # To avoid load_rawsets + conversion each time
    # valiset = Trainset.load(filename)

    if new_model:
        '''Flow creation'''
        flow = CBCEstimator(params_list, flow_config, net_config, name=name,
                            workdir=outdir, mode='net+flow', preprocess=pre_process)
    else:
        '''Training continuation of previous model'''
        flow = CBCEstimator.load_from_file(outdir / f'{name}.pt')
        flow.rename(name)  # new name if preferred



    dataset = load_rawsets_pool(rawdat_dir, seeds2names(seeds, zero_pad=zero_pad), processes=os.cpu_count())
    dataset = convert_dataset(dataset, params_list, name=dataset_name)
    # dataset.save(filename)  # To avoid load_rawsets + conversion each time
    # dataset = Trainset.load(filename)


    # flow.turn_net_grad('off')  # If at some point we want to only train the flow
    flow.train(dataset,
               outdir,
               train_config,
               valiset, )
    flow.save_to_file(f'{flow.name}.pt')  # Saves it in the workdir
