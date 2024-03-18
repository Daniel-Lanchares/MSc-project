import time
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
        self._df.name = name

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

    @classmethod
    def concat(cls, trainsets: list, name: str = None, *args, **kwargs):
        if name is None:
            name = trainsets[0].name
        dfs = [tset._df for tset in trainsets]
        return TrainSet(data=pd.concat(dfs, *args, **kwargs), name=name)

    @classmethod
    def load(cls, path: str | Path | list[str] | list[Path], name: str = None, verbose: bool = True):
        # Trainset loads and saves its internal DataFrame, not the TrainSet itself
        if type(path) is list:
            return TrainSet.concat([TrainSet.load(_path) for _path in path], name=name)
        try:
            _name, df = torch.load(path)
        except ValueError:
            # Then is a lightweight save
            df = pd.read_pickle(path)
            _name = path.parts[-1][:-3]

        if name is None:
            name = _name

        if verbose:
            print(f'Loaded TrainSet {_name}')
        return TrainSet(data=df, name=name)

    def save(self, path, lightweight: bool = False):
        # Trainset loads and saves its internal DataFrame, not the TrainSet itself
        # lightweight saves the dataframe contents only. loses the name but is much more compact
        # If it isn't given a pickle file it will name it after itself
        if path.parts[-1][-3:] != '.pt':
            path = path / f'{self.name}.pt'
        if lightweight:
            self._df.to_pickle(path)
        else:
            torch.save((self.name, self._df), path)

    def __len__(self):
        return self._df.shape[0]

    def __repr__(self):
        return self._df.__repr__()


class FeederDataset(Dataset):
    """
    Object to be fed to the DataLoader. It implements basic functionality of feeding system
    and ensures that the model is given a torch.Tensor and not a DataFrame or an array.
    """

    def __init__(self, trainset: TrainSet | str | Path):
        # dataset can either be a path to a dataset or the dataset itself
        if isinstance(trainset, str | Path):
            trainset = torch.load(trainset)

        # print([type(tens)for tens in trainset['labels']])

        labels = torch.cat([torch.reshape(tens, (1, len(tens))) for tens in trainset['labels']])

        self.x, self.y = (torch.cat(list(trainset.values[:, 0])), labels)

    def __len__(self):
        return self.x.shape[0]  # Number of images

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it is callable call it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)


def _check_weight_viability(model, data, max_val, max_iter, counter: int = 1):
    x, y = data
    loss_value = -model.log_prob(inputs=y.float(), context=x).mean()
    print(f'Viability test {counter}')
    print(f'Initial loss: {loss_value.item():.4}')
    if loss_value < max_val:
        print(f'Loss under {max_val}. Test passed')
        return
    else:
        print(f'Loss over {max_val}. Test failed')
        if counter <= max_iter:
            print('Trying again')
            reset_all_weights(model)
            counter += 1
            del loss_value  # Might be causing memory issues
            time.sleep(2)  # Gives time to read if someone is taking a look, won't affect hours long training
            _check_weight_viability(model, data, max_val, max_iter, counter)
            return
        else:
            print(f'Reached {max_iter} iterations without success. Training from current weights')
            return


def check_weight_viability(model, dataloader, max_val: float = None, max_iter: int = None) -> None:
    if max_val is None:
        max_val = 1e4
    if max_iter is None:
        max_iter = 20
    for data in dataloader:
        # Not really designed to loop
        # Hacky way of doing it, but torch's DataLoader doesn't define other 'proper' way of accessing items
        _check_weight_viability(model, data, max_val, max_iter)
        return


def train_model(model, dataloader, train_config, valiloader):
    n_epochs = train_config['num_epochs']
    lr = train_config['learning_rate']
    opt = opt_dict[train_config['optim_type']](model.parameters(), lr)

    if 'sched_kwargs' in train_config and train_config['sched_kwargs'] is not None:
        sched_type = train_config['sched_kwargs'].pop('type')
        sched = sched_dict[sched_type](opt, **(train_config['sched_kwargs']))
    else:
        sched = None

    if 'checkpoint_every_x_epochs' in train_config and train_config['checkpoint_every_x_epochs'] is not None:
        checkpt = train_config['checkpoint_every_x_epochs']

    # Train model
    losses = []
    epochs = []
    vali_losses = []
    vali_epochs = []
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1}')
        n_batches = len(dataloader)
        batch_epochs = []
        batch_losses = []
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
            batch_epochs.append(epoch + i / n_batches)
            batch_losses.append(loss_value.item())

        epochs.append(batch_epochs)
        losses.append(batch_losses)

        temp_loss = np.array(losses).mean(axis=1)
        if epoch > 0:
            print(f'\nAverage train: {temp_loss[-1]:.6}, Delta: {(temp_loss[-1] - temp_loss[-2]):.6} '
                  f'({(temp_loss[-1] - temp_loss[-2]) / temp_loss[-2] * 100:.6}%)\n')
        else:
            print(f'\nAverage train: {temp_loss[0]:.6}\n')

        n_batches = len(valiloader)
        if valiloader is not None:
            print(f'Validation of epoch {epoch + 1:3d}')
            batch_epochs = []
            batch_losses = []
            for i, (x, y) in enumerate(valiloader):
                loss_value = -model.log_prob(inputs=y.float(), context=x).mean()

                print(f'Epoch {epoch + 1:3d}, batch {i:3d}: {loss_value.item():.4}')
                batch_epochs.append(epoch + i / n_batches)
                batch_losses.append(loss_value.item())

            vali_epochs.append(batch_epochs)
            vali_losses.append(batch_losses)

            temp_loss = np.array(vali_losses).mean(axis=1)
            if epoch > 0:
                print(f'\nAverage valid: {temp_loss[-1]:.6}, Delta: {(temp_loss[-1] - temp_loss[-2]):.6} '
                      f'({(temp_loss[-1] - temp_loss[-2]) / temp_loss[-2] * 100:.6}%)\n')
            else:
                print(f'\nAverage valid: {temp_loss[0]:.6}\n')

        # Update Scheduler
        if sched is not None:
            sched.step()  # TODO Implement possibility for schedulers that take loss values. Not urgent

    # For manually variable lr
    # for g in optim.param_groups:
    #     g['lr'] = 0.001

    return np.array(epochs), np.array(losses), np.array(vali_epochs), np.array(vali_losses)
