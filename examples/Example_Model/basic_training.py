# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import models


from CBC_estimator.training.train_utils import QTDataset, train_model
from CBC_estimator.nn.net_utils import create_feature_extractor
from CBC_estimator.nn.flow_utils import create_flow # Only testing the ResNet for now
from CBC_estimator.dataset.dataset_utils import convert_dataset

dataset_dir = Path('C:/Users/danie/OneDrive/Escritorio/Física/5º (Máster)/TFM/Scripts/Datasets/11 parameters') # Right now set for aligned-spins
trainset_dir = Path('C:/Users/danie/OneDrive/Escritorio/Física/5º (Máster)/TFM/Scripts/MSc project/examples/Trainsets')
trainset_dir = Path('../Trainsets')
# print(dataset_dir)
# print(trainset_dir.exists()) # pathhlib says it exists, torch.save() claims otherwise

params_list = [
    'chirp_mass',
    'chi_eff',
    'd_L',
    'NAP'
    ]

dataset = []
for seed in range(1):
    dataset = np.concatenate((dataset, torch.load(dataset_dir/f'Raw_dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)#trainset_dir/'4_parameter_trainset.pt')

# All parameters are merelly examples
train_config = {
    'num_epochs': 20,
    'checkpoint_every_x_epochs': 5,
    'batch_size': 64,
    'optim_type': 'SGD', # 'Adam'
    'learning_rate': 0.001, #Study how to change it mid-training
    }

net_config = { # This are to create a net from scratch
    'input_channels': 3,
    'output_length': 4, #4 when not testing the flow 
    'blocks_sizes': np.array([64, 128, 256, 512]),
    'deepths': [2, 2, 2, 2]
    }
extractor_config = { # This are in substitution of the previous
                     # Both are shown at once merely as an example
    'n_features': 128,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }
flow_config = { # As it is now, it explodes instantly
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,
    'base_transform_kwargs': { #This I will study in more detail #TODO
        'hidden_dim': 64,
        'num_transform_blocks': 2,
        'num_bins': 8
        },
    }


traindir = 'training_test_3_(extractor+flow,lr=0.001)'
train_dataset = QTDataset(trainset)
train_dataloader = DataLoader(train_dataset, batch_size=train_config['batch_size'])


# model = EmbeddedFlow(net_kwargs=net_config, 
#                      flow_kwargs=flow_config)

# Perhaps this way may be more readable

net = create_feature_extractor(**extractor_config)

# Sanity check
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0: 
#         print(x.shape)
#         print(net(x))

model = create_flow(emb_net=net, **flow_config)
# print(model)
# model = net

print(model)
fig = plt.figure()
plt.show()

# Sanity check II
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0: 
#         print(x.shape)
#         print(model.sample(1, context=x))
#         print(model.log_prob(y.float(),x)) # RuntimeError: expected scalar type Double but found Float FIXED


# Exploding gradient. And not just a TNT explosion, quite the nuke
# Seems to get better with bigger datasets. need to generate more
epoch_data, loss_data = train_model(model, train_dataloader, traindir, train_config)


#Average over batches: 1 loss per epoch
epoch_data_avgd = epoch_data.reshape(20,-1).mean(axis=1)
loss_data_avgd = loss_data.reshape(20,-1).mean(axis=1)

plt.figure(figsize=(10, 8))
plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
plt.xlabel('Epoch Number')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error (avgd per epoch)')
plt.show()

# model.load_state_dict(torch.load(Path(traindir)/'Model_state_dict.pt'))
# for i, (x,y) in enumerate(train_dataloader):
#     if i==0:
#         print(model.sample(6, x))
#         print(y)