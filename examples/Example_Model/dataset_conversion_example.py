# -*- coding: utf-8 -*-
import torch

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint



# Basic example for dataset conversion and study


from CBC_estimator.dataset.dataset_utils import convert_dataset, plot_hists, image

# This way I can have Toy, test and proper datasets in separate folders
files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
dat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'

# Dataset would be converted according to training parameters

params_list = [
    'chirp_mass',
    'chi_eff',
    'd_L',
    'NAP'
    ]
# # weights='ResNet18_Weights.DEFAULT' 
# resnet18 = models.resnet18()
# resnet18.load_state_dict(torch.load('Pre_ResNet18_state_dict.pt'))
# print(resnet18)

dataset = []
for seed in range(1):
    dataset = np.concatenate((dataset, torch.load(dat_dir/f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

# print(np.array([data['id'] for data in dataset]))
print(f'Dataset size: {len(dataset)}\n')
#pprint(dataset[0]['parameters'])

trainset = convert_dataset(dataset, params_list)

#TODO Make funtion in dataset_utils that plots the image of an iterable of indices
# n = 4
# pprint(dataset[n]['SNR'])
# plt.imshow(image(dataset[n]))
# plt.show()

# print('Raw dataset')
# pprint(dataset[0])
# print()
# print('4 parameter trainset')
# print('images')
# print(trainset[0][0])
# print('labels')
# print(trainset[1][0])

# Add funtion to dataset_utils that does this more pretty (get_unit(parameter), etc...)
# plt.hist(extract_parameters(dataset, ['mass_1','mass_2']))
layout = np.ones((2,2), dtype=object)
layout[0,0] = ['chi_eff']#['mass_1', 'mass_2']
layout[1,0] = ['chirp_mass']
layout[0,1] = ['NAP']
layout[1,1] = ['d_L']
fig = plot_hists(dataset, layout, figsize=(10, 8), bins=10)
plt.show()

