# -*- coding: utf-8 -*-
import torch

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Basic example for dataset conversion and study


from CBC_estimator.core.conversion_utils import convert_dataset, plot_hists, plot_images

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


layout = np.array([i for i in range(30)])
layout.shape = (5, 6)


trainset = convert_dataset(dataset, params_list)

# This discussion could be used to visualise all images on a dataset through pages
# https://stackoverflow.com/questions/33139496/how-to-plot-several-graphs-and-make-use-of-the-navigation-button-in-matplotlib

fig = plot_images(dataset, layout, figsize=(14, 10),
                  title_maker=lambda x: f'{x["id"]}')
fig.suptitle('Q-Transform images (RGB = (L1, H1, V1))')
plt.tight_layout()

layout = np.array([30+i for i in range(30)])
layout.shape = (5, 6)
fig2 = plot_images(dataset, layout, figsize=(14, 10),
                   title_maker=lambda x: f'{x["id"]}')
fig2.suptitle('Q-Transform images (RGB = (L1, H1, V1))')
plt.tight_layout()
plt.show()

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
layout = np.ones((2, 2), dtype=object)
layout[0, 0] = ['chi_eff']  # ['mass_1', 'mass_2']
layout[1, 0] = ['chirp_mass']
layout[0, 1] = ['NAP']
layout[1, 1] = ['d_L']
fig = plot_hists(dataset, layout, figsize=(10, 8), bins=10)
plt.show()

