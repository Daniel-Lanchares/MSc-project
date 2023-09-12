
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import models

from dtempest.core.net_utils import create_feature_extractor
from dtempest.core.flow_utils import create_flow
from dtempest.core.conversion_utils import convert_dataset, plot_images

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Traindir'
traindir = train_dir / 'training_test_2_(trained over 0,lr=0.05)'

params_list = [
    'chirp_mass',
    # 'chi_eff',
    # 'd_L',
    # 'NAP'
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

model = create_flow(emb_net=create_feature_extractor(**extractor_config), **flow_config)

model.load_state_dict(torch.load(traindir / 'Model_state_dict.pt'))
model.eval()

dataset = []
for seed in range(1):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir/f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

trainset = convert_dataset(dataset, params_list)#trainset_dir/'4_parameter_trainset.pt')

pre_process = models.ResNet18_Weights.DEFAULT.transforms(antialias=True)  # True for internal compatibility reasons
# print(pre_process)

processed_trainset = (pre_process(trainset[0]), trainset[1])  # May implement directly in convert_dataset
del trainset  # The least RAM used the better

n = 16
print(f'Label: {processed_trainset[1][n]}')
image = processed_trainset[0][n].expand(1, 3, 224, 224)
print(type(image), image.shape)
samples, logprob = model.sample_and_log_prob(num_samples=5000, context=image)
print(f'Prediction: {samples.detach()[0,2,0]}')
# print(f'Log Prob: {logprob.detach()}')

'''
It is surprisingly good for a lot of images. 
Should make a function to plot and get confidence intervals (probably based on bilby & corner)
'''

layout = np.array([x for x in range(n, n+4)])
layout.shape = (2, 2)
fig = plot_images(dataset, layout)

fig2 = plt.figure()
plt.hist(samples.detach().flatten(), bins=20)
plt.axvline(processed_trainset[1][n][0], color='k', label='chirp_mass')
plt.show()