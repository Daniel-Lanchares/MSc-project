# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:31:38 2023

@author: danie
"""
import torch

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint


'''

 Q-Transform          Extracted Parameters         Unconditional Flow?
(128, 128, 3)  -->      (n_parameters,)    -->   (n_samples, n_parameters)

'''

# Ejemplo de script para crear y entrenar el modelo


from CBC_estimator.dataset.dataset_utils import convert_dataset, plot_hists

# This way I can have Toy, test and proper datasets in separate folders
dat_dir = Path('C:/Users/danie/OneDrive/Escritorio/Física/5º (Máster)/TFM/Scripts/Datasets') 
print(dat_dir.resolve())

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
    dataset = np.concatenate((dataset, torch.load(dat_dir/f'Raw_dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

# print(np.array([data['id'] for data in dataset]))
print(f'Dataset size: {len(dataset)}\n')
#pprint(dataset[0]['parameters'])

trainset = convert_dataset(dataset, params_list)

print('Raw dataset')
pprint(dataset[0])
print()
print('4 parameter trainset')
print('images')
print(trainset[0][0])
print('labels')
print(trainset[1][0])

# Add funtion to dataset_utils that does this more pretty (get_unit(parameter), etc...)
# plt.hist(extract_parameters(dataset, ['mass_1','mass_2']))
layout = np.ones((2,2), dtype=object)
layout[0,0] = ['chi_eff']#['mass_1', 'mass_2']
layout[1,0] = ['chirp_mass']
layout[0,1] = ['NAP']
layout[1,1] = ['d_L']
fig = plot_hists(dataset, layout, figsize=(10, 8), bins=10)
plt.show()

