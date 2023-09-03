# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import models


from CBC_estimator.core.train_utils import QTDataset, train_model
from CBC_estimator.core.net_utils import create_feature_extractor
from CBC_estimator.core.flow_utils import create_flow
from CBC_estimator.core.conversion_utils import convert_dataset

# dataset_dir = Path('C:/Users/danie/OneDrive/Escritorio/Física/5º (Máster)/TFM/Scripts/Datasets/11 parameters') # Right now set for aligned-spins
# trainset_dir = Path('C:/Users/danie/OneDrive/Escritorio/Física/5º (Máster)/TFM/Scripts/MSc project/examples/Trainsets')
# trainset_dir = Path('../Trainsets')
files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Traindir'


params_list = [
    'chirp_mass',
    # 'chi_eff',
    # 'd_L',
    # 'NAP'
    ]

dataset = []
for seed in range(2):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir/f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)#trainset_dir/'4_parameter_trainset.pt')

del dataset

# All parameters are merelly examples
train_config = {
    'num_epochs': 20,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 64,
    'optim_type': 'SGD',  # 'Adam'
    'learning_rate': 0.005,  # Study how to change it mid-training
    }

net_config = {  # These are to create a net from scratch
    'input_channels': 3,
    'output_length': 4,  # 4 when not testing the flow
    'blocks_sizes': np.array([64, 128, 256, 512]),
    'deepths': [2, 2, 2, 2]
    }
extractor_config = {  # These are in substitution of the previous
                      # Both are shown at once merely as an example
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


pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons
# print(pre_process)

processed_trainset = (pre_process(trainset[0]), trainset[1])  # May implement directly in convert_dataset
print('preprocessed_trainset')


del trainset  # The least RAM used the better

# Test for prepocess (note that I did not correct the normalization, so it shows clipped)
# n=0
# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(1,2,1)
# ax1.imshow(image(trainset[0][n]))
# ax2 = fig.add_subplot(1,2,2)
# ax2.imshow(image(processed_trainset[0][n]))
# plt.show()

traindir = train_dir / 'training_test_3_(trained over 2,lr=0.005)'
train_dataset = QTDataset(processed_trainset)
train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'])

del processed_trainset


# model = EmbeddedFlow(net_kwargs=net_config, 
#                      flow_kwargs=flow_config)

# Perhaps this way may be more readable

net = create_feature_extractor(**extractor_config)

# Sanity check
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0: 
#         print(x.shape)
#         print(net(x))

# print(model)
# model = net

traindir0 = train_dir / 'training_test_2_(trained over 0,lr=0.05)'

model = create_flow(emb_net=net, **flow_config)
model.load_state_dict(torch.load(traindir0 / 'Model_state_dict.pt'))

# print(model)


# Sanity check II
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0: 
#         print(x.shape)
#         print(model.sample(1, context=x))
#         print(model.log_prob(y.float(),x)) # RuntimeError: expected scalar type Double but found Float FIXED


# Exploding gradient. And not just a TNT explosion, quite the nuke
# Seems to get better with bigger datasets. need to generate more
epoch_data, loss_data = train_model(model, train_dataloader, traindir, train_config)


# Average over batches: 1 loss per epoch
epoch_data_avgd = epoch_data.reshape(20, -1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20, -1).mean(axis=1)

plt.figure(figsize=(10, 8))
plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error (avgd per epoch)')
plt.savefig(traindir/'loss_plot.png', format='png')
plt.show()

# model.load_state_dict(torch.load(Path(traindir)/'Model_state_dict.pt'))
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0:
#         print(model.sample(6, x))
#         print(y)