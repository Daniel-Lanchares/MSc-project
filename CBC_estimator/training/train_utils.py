# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 18:24:05 2023

@author: danie
"""
import os
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

opt_dict = {'SGD': SGD, 'Adam': Adam}
loss_dict = {'MSE': nn.MSELoss, 'CE': nn.CrossEntropyLoss}


class QTDataset(Dataset):

    def __init__(self, dataset):
        # dataset can either be a path to a dataset or the dataset itself
        if isinstance(dataset, str) or isinstance(dataset, Path):
            dataset = torch.load(dataset)
        self.x, self.y = dataset

    def __len__(self):
        return self.x.shape[0]  # Number of images

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


def train_model(model, dataloader, outdir, train_config):
    # Make outdir if it doesn't exist
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outdir = Path(outdir)

    n_epochs = train_config['num_epochs']
    chckpt = train_config['checkpoint_every_x_epochs']
    lr = train_config['learning_rate']
    opt = opt_dict[train_config['optim_type']](model.parameters(), lr)

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        N = len(dataloader)
        for i, (x, y) in enumerate(dataloader):
            # Update the weights of the network
            opt.zero_grad()
            loss_value = -model.log_prob(inputs=y.float(), context=x).mean()
            # L = nn.MSELoss()
            # loss_value = L(model(x), y.float())  # To test just the net -> Here lies the gradient issue
            print(loss_value.item())
            loss_value.backward()
            opt.step()
            # Store training data
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())

    torch.save(model.state_dict(), outdir / 'Model_state_dict.pt')
    return np.array(epochs), np.array(losses)
