from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from dtempest.core import Estimator
from dtempest.core.conversion_utils import convert_dataset, make_image

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '1: Chirp mass estimator'
traindir = train_dir / 'training_test_1'

dataset = []
for seed in range(3, 4):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir / f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

flow = Estimator.load_from_file(traindir / 'v0.1.5.pt')

n = 8
trainset = convert_dataset(dataset, flow.param_list, name='Dataset 3')


p_trainset = flow.preprocess(deepcopy(trainset))

# https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573

# we will save the conv layer weights in this list '''Don't know why, though'''
model_weights = []
# we will save the 49 conv layers in this list
conv_layers = []
# get all the model children as list
model_children = list(flow.model._embedding_net.children())
# counter to keep count of the conv layers
counter = 0
# append all the conv layers and their respective wights to the list
for i in range(len(model_children)):
    if type(model_children[i]) == nn.Conv2d:
        counter += 1
        model_weights.append(model_children[i].weight)
        conv_layers.append(model_children[i])
    elif type(model_children[i]) == nn.Sequential:
        for j in range(len(model_children[i])):
            for child in model_children[i][j].children():
                if type(child) == nn.Conv2d:
                    counter += 1
                    model_weights.append(child.weight)
                    conv_layers.append(child)
print(f"Total convolution layers: {counter}")
print("conv_layers")

image = p_trainset['images'][n]

outputs = []
names = []
for layer in conv_layers[0:]:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
# print(len(outputs))
# for feature_map in outputs:
#     print(feature_map.shape)

processed = []
for feature_map in outputs:
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]
    processed.append(gray_scale.data.cpu().numpy())
# for fm in processed:
#     print(fm.shape)

fontsize = 30
fig = plt.figure(figsize=(50, 30))
ax = fig.add_subplot(4, 5, 1)
plt.imshow(make_image(trainset['images'][n]))
ax.axis("off")
ax.set_title(f'Original: {dataset[n]["id"]}', fontsize=fontsize)

ax = fig.add_subplot(4, 5, 2)
plt.imshow(make_image(p_trainset['images'][n].squeeze(0)))
ax.axis("off")
ax.set_title('Preprocessed (clipped)', fontsize=fontsize)

for i in range(len(processed)):
    a = fig.add_subplot(4, 5, i+1+2)
    imgplot = plt.imshow(processed[i])
    a.axis("off")
    a.set_title(names[i].split('(')[0]+f' {i+1}', fontsize=fontsize)

plt.savefig(str('feature_maps.jpg'), bbox_inches='tight')
