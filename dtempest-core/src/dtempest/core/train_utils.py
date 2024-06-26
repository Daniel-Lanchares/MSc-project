import time
from pathlib import Path
from typing import get_type_hints
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR, StepLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

from dtempest.core.common_utils import RawSet

sched_dict = {'Plateau': ReduceLROnPlateau,
              'StepLR': StepLR,
              'MultiStepLR': MultiStepLR,
              'cosine': CosineAnnealingLR}
opt_dict = {'SGD': SGD, 'Adam': Adam}
loss_dict = {'MSE': nn.MSELoss, 'CE': nn.CrossEntropyLoss}


class TrainSet:
    """
    Wrapper class for pandas DataFrame. Allows me to name it.
    When applying any funtion that returns knows what to return
     (Array-like objects are not easily extended or subclassed)
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
        elif hasattr(self._df, attr):
            # First let's deal with some special, often used, cases
            if attr == 'T':
                return self.transpose()
            if attr == 'sample':
                return Overarcher(getattr(self._df, attr), self)

            # This allows it to transform the internal dataframe and return a new TrainSet... when it works
            try:
                if hasattr(type(self._df), attr) and hasattr(getattr(self._df, attr), '__annotations__'):
                    returns = getattr(self._df, attr).__annotations__['return'].split(' | ')
                    # TODO: consider exceptions raised from methods that can return different types based on arguments
                    if 'DataFrame' in returns:
                        return Overarcher(getattr(self._df, attr), self)
                    elif 'Series' in returns:
                        # Not doing anything with it yet, but interesting for other classes
                        return getattr(self._df, attr)
                    elif 'str' in returns:
                        return getattr(self._df, attr)  # Not doing anything fancy, but can catch formatting calls
                    else:
                        return getattr(self._df, attr)
                else:
                    return getattr(self._df, attr)
            except KeyError:
                return getattr(self._df, attr)
            # return getattr(self._df, attr)
        else:
            raise AttributeError(f"{type(self)} has no attribute '{attr}'")

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

    def save(self, path: str | Path, lightweight: bool = False):
        # Trainset loads and saves its internal DataFrame, not the TrainSet itself
        # lightweight saves the dataframe contents only. loses the name but is much more compact
        # If it isn't given a pickle file it will name it after itself
        if type(path) is str:
            path = Path(path)
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


class Overarcher:
    def __init__(self, method, wrapper_object: TrainSet):
        self.method = method
        self.wrapper = wrapper_object

    def __call__(self, *args, **kwargs):
        return type(self.wrapper)(data=self.method(*args, **kwargs), name=self.wrapper.name)


def check_trainset_format(trainset: TrainSet | str | Path) -> TrainSet:
    """
        Allows functions to be given either a dataset tensor or its path.

        Parameters
        ----------
        trainset : str, pathlib.Path or TrainSet.

        Returns
        -------
        dataset : TrainSet.

        """
    if isinstance(trainset, str | Path):
        trainset = TrainSet.load(trainset)
    return trainset


class FeederDataset(Dataset):
    """
    Object to be fed to the DataLoader. It implements basic functionality of feeding system
    and ensures that the model is given a torch.Tensor and not a DataFrame or an array.
    """

    def __init__(self, trainset: TrainSet | str | Path):
        # dataset can either be a path to a dataset or the dataset itself
        if isinstance(trainset, str | Path):
            trainset = torch.load(trainset)

        if trainset['labels'].iloc[0].dim() != 0:
            labels = torch.cat([torch.reshape(tens, (1, len(tens))) for tens in trainset['labels']])
        else:   # Deals with 1 parameter models
            labels = torch.cat([torch.reshape(tens, (1, 1)) for tens in trainset['labels']])

        self.x, self.y = (torch.cat(list(trainset.values[:, 0])), labels)

    def __len__(self):
        return self.x.shape[0]  # Number of images

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


# count = 0
def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    # global count

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # global count
        # - check if the current module has reset_parameters & if it is callable call it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        # if reset_parameters is not None:
        # count += 1
        if callable(reset_parameters):
            m.reset_parameters()
        # if isinstance(m, nn.Conv2d):  # This should target the embedding net
        #     print(m)
        #     print(m.weight.data)
        #     torch.nn.init.xavier_uniform_(m.weight.data)

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)
    # print(f'number of components with reset: {count}')
    # count = 0


def _check_weight_viability(model, data, max_val, max_iter, counter: int = 1, tenfold_at_limit: bool | int = False):
    if type(tenfold_at_limit) is bool:
        tenfold_at_limit = int(not tenfold_at_limit)
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
            time.sleep(1)  # Gives time to read if someone is taking a look, won't affect hours long training
            _check_weight_viability(model, data, max_val, max_iter, counter, tenfold_at_limit)
            return
        else:
            if 0 < tenfold_at_limit < 10:  # Keeps track of how many times it doubles max value, shouldn't be hardcoded
                print(f'Doubling maximum initial loss to {10*max_val} and trying again')
                counter = 0
                tenfold_at_limit += 1
                _check_weight_viability(model, data, 10*max_val, max_iter, counter, tenfold_at_limit)
                return

            print(f'Reached {max_iter} iterations without success. Training from current weights')
            return


def check_weight_viability(model, dataloader, max_val: float = None, max_iter: int = None,
                           double_at_limit: bool = False) -> None:
    if max_val is None:
        max_val = 1e4
    if max_iter is None:
        max_iter = 20
    elif max_iter == 0:
        return
    for data in dataloader:
        # Not really designed to loop
        # Hacky way of doing it, but torch's DataLoader doesn't define other 'proper' way of accessing items
        _check_weight_viability(model, data, max_val, max_iter, double_at_limit)
        return


def special_weight_check(model, trainset, batch_size: int, max_val: float = None, max_iter: int = None,
                         counter: int = 0) -> TrainSet:
    if max_val is None:
        max_val = 1e4
    if max_iter is None:
        max_iter = 20
    elif max_iter == 0:
        return trainset

    dataloader = DataLoader(FeederDataset(trainset), batch_size=batch_size)
    for data in dataloader:
        # Not really designed to loop
        # Hacky way of doing it, but torch's DataLoader doesn't define other 'proper' way of accessing items
        x, y = data
        loss_value = -model.log_prob(inputs=y.float(), context=x).mean()
        print(f'Viability test {counter}')
        print(f'Initial loss: {loss_value.item():.4}')
        if loss_value < max_val:
            print(f'Loss under {max_val}. Test passed')
            return trainset
        else:
            print(f'Loss over {max_val}. Test failed')
            if counter <= max_iter:
                print('Trying again')
                trainset = trainset.sample(frac=1, random_state=counter)
                counter += 1
                del loss_value  # Might be causing memory issues
                time.sleep(1)  # Gives time to read if someone is taking a look, won't affect hours long training
                trainset = special_weight_check(model, trainset, batch_size, max_val, max_iter, counter)
                return trainset
            else:
                print(f'Reached {max_iter} iterations without success. Training from current weights')
                return trainset


def loss_print(epoch: int, losses: list, code='train', fmt='.3'):
    temp_loss = np.array(losses).mean(axis=1)
    deviation = np.array(losses).std(axis=1)
    if epoch > 0:
        msg = (f'\nAverage {code}: {temp_loss[-1]:{fmt}}±{deviation[-1]:{fmt}}, '
               f'Delta: {(temp_loss[-1] - temp_loss[-2]):{fmt}} '
               f'({(temp_loss[-1] - temp_loss[-2]) / temp_loss[-2] * 100:{fmt}}%)\n')
    else:
        msg = f'\nAverage {code}: {temp_loss[0]:{fmt}}±{deviation[0]:{fmt}}\n'
    # Consider logging to file
    print(msg)


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

        loss_print(epoch, losses, code='train')

        if valiloader is not None:
            n_batches = len(valiloader)
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

            loss_print(epoch, vali_losses, code='valid')

        # Update Scheduler
        if sched is not None:
            sched.step()  # TODO Implement possibility for schedulers that take loss values. Not urgent

    # For manually variable lr
    # for g in optim.param_groups:
    #     g['lr'] = 0.001

    return np.array(epochs), np.array(losses), np.array(vali_epochs), np.array(vali_losses)
