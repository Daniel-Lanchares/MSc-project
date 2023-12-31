import os
from pathlib import Path
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR
from torch.utils.data import Dataset

from dtempest.core.common_utils import RawSet

sched_dict = {'Plateau': ReduceLROnPlateau, 'StepLR': StepLR, 'MultiStepLR': MultiStepLR}
opt_dict = {'SGD': SGD, 'Adam': Adam}
loss_dict = {'MSE': nn.MSELoss, 'CE': nn.CrossEntropyLoss}


class TrainSet:
    """
    Wrapper class for pandas DataFrame. Allows me to name it.
    When applying any funtion that returns a copy remember to recreate
    the object (Array-like objects are not easily extended or subclassed)
    """

    def __init__(self, name: str = None, *args, **kwargs):

        if 'data' in kwargs.keys() and isinstance(kwargs['data'], pd.DataFrame):
            # Initialize from a DataFrame
            self._df = kwargs['data']
        else:
            # Initialize from a pandas-recognized data structure
            self._df = pd.DataFrame(*args, **kwargs)
        if name is None:
            # If an explicit name is not given, take it from the data...
            for arg in args:
                if isinstance(arg, RawSet):
                    name = arg.name
                    break
            else:
                # ... Or have a generic name
                name = type(self).__name__
        self.name = name

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        if isinstance(self._df[item], pd.DataFrame):
            return type(self)(data=self._df[item], name=self.name)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return self._df.__repr__()


class QTDataset(Dataset):
    """
    Object to be fed to the DataLoader. It implements basic functionality of feeding system
    and ensures that the model is given a torch.Tensor and not a DataFrame or an array.
    """

    def __init__(self, trainset: TrainSet | str | Path):
        # dataset can either be a path to a dataset or the dataset itself
        if isinstance(trainset, str | Path):
            trainset = torch.load(trainset)

        labels = torch.cat([torch.reshape(tens, (1, len(tens))) for tens in trainset['labels']])

        self.x, self.y = (torch.cat(list(trainset.values[:, 0])), labels)

    def __len__(self):
        return self.x.shape[0]  # Number of images

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


def train_model(model, dataloader, train_config):

    n_epochs = train_config['num_epochs']
    checkpt = train_config['checkpoint_every_x_epochs']
    lr = train_config['learning_rate']
    opt = opt_dict[train_config['optim_type']](model.parameters(), lr)

    if 'sched_kwargs' in train_config and train_config['sched_kwargs'] is not None:
        sched_type = train_config['sched_kwargs'].pop('type')
        sched = sched_dict[sched_type](opt, **(train_config['sched_kwargs']))
    else:
        sched = None

    # Train model
    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}')
        N = len(dataloader)
        for i, (x, y) in enumerate(dataloader):
            # Update the weights of the network
            loss_value = -model.log_prob(inputs=y.float(), context=x).mean()
            print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {loss_value.item():.4}')

            loss_value.backward()
            if 'grad_clip' in train_config and train_config['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_config['grad_clip'])
            opt.step()
            opt.zero_grad()
            # Store training data
            epochs.append(epoch + i / N)
            losses.append(loss_value.item())

        temp_loss = np.array(losses).reshape(epoch + 1, -1).mean(axis=1)
        if epoch > 0:
            print(f'\nAverage: {temp_loss[-1]:.6}, Delta: {(temp_loss[-1] - temp_loss[-2]):.6} '
                  f'({(temp_loss[-1] - temp_loss[-2]) / temp_loss[-2] * 100:.6}%)\n')
        else:
            print(f'\nAverage: {temp_loss[0]:.6}\n')

        # Update Scheduler
        if sched is None:
            continue
        else:
            sched.step()  # TODO Implement possibility for schedulers that take loss values



    # For manually variable lr
    # for g in optim.param_groups:
    #     g['lr'] = 0.001
    return np.array(epochs), np.array(losses)
