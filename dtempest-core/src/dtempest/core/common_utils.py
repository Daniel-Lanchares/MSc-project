import os
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from pathlib import Path


# Look for more colors just in case
class PrintStyle:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'


def identity(x):
    return x


def check_format(dataset):
    """
    Allows functions to be given either a dataset tensor or its path.

    Parameters
    ----------
    dataset : str, pathlib.Path or torch.tensor.

    Returns
    -------
    dataset : torch.tensor.

    """
    if isinstance(dataset, (str, Path)):
        dataset = torch.load(dataset)
    return dataset


def get_missing_args(type_excep):
    crumbs = type_excep.args[0].split(' ')
    n_args = int(crumbs[2])

    return [crumbs[-2 * i + 1][1:-1] for i in reversed(range(1, n_args + 1))]


def get_extractor(name):
    import torchvision.models as models
    model = getattr(models, name)
    weights = getattr(getattr(models, f'{models_dict[name]}_Weights'), 'DEFAULT')
    pre_process = weights.transforms(antialias=True)  # True for internal compatibility reasons

    return model, weights, pre_process


models_dict = {
    'resnet18': 'ResNet18'
}


class RawSet:  # TODO
    """
    Iterable of OrderedDicts containing fully described events (injections in GW)
    Could be a subclass of list...
    """

    def __init__(self, rawdata, name: str = None, *args, **kwargs):
        self._df = pd.Series(*args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name

        for data in rawdata:
            if 'id' in data:
                self[data['id']] = data
            else:
                self[len(self._df)] = data  # Untested

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        # if isinstance(self._df[item], pd.Series):  # Need to study whether I will actually use this
        #     return type(self)(rawdata=self._df[item], name=self.name)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return self._df.__repr__()

    def change_parameter_name(self, old_key, to):  # to: New key
        for data in self:
            # data['parameters']['luminosity_distance'] = data['parameters'].pop('d_L')
            data['parameters'] = OrderedDict(
                [(to, v) if k == old_key else (k, v) for k, v in data['parameters'].items()])


def seeds2names(seeds):
    if not hasattr(seeds, '__iter__'):
        seeds = [seeds, ]
    return [f'Raw_Dataset_{seed:03}.pt' for seed in seeds]


def load_rawsets(directory, names: list[str], verbose: bool = True):
    if not hasattr(names, '__iter__'):
        names = [names, ]
    dataset = []
    for name in names:
        dataset = np.concatenate((dataset, torch.load(directory / name)))
        if verbose:
            print(f'Loaded {name}')
    return RawSet(dataset, name='+'.join(names))


def load_losses(directory: str | Path,
                model: str = None,
                stages: int | list[int] = None,
                validation: bool = False,
                verbose: bool = True) -> tuple[np.array, np.array] | tuple[np.array, np.array, np.array]:

    subdirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    if stages is None:
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs}
    else:
        if not hasattr(stages, '__iter__'):
            stages = [stages, ]
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs if int(subdir.split('_')[-1]) in stages}

    if model is not None:
        chosen_subs = {stage: subdir for stage, subdir in chosen_subs.items() if int(subdir.split('_')[-3]) == model}

    epochs = []
    losses = []
    validations = np.array([])
    for i, (stage, sub) in enumerate(chosen_subs.items()):
        epoch, loss = torch.load(Path(directory) / sub / 'loss_data.pt')
        if i == 0:
            epochs = np.concatenate((epochs, epoch))

        else:
            epochs = np.concatenate((epochs, epoch + epochs[-1]*np.ones_like(epoch)))

        losses = np.concatenate((losses, loss))
        if validation:
            try:
                validations = np.concatenate((validations, torch.load(Path(directory) / sub / 'validation_data.pt')[1]))
            except FileNotFoundError as exc:
                print(f'Validation not found in {sub}')
                print(exc)
        if verbose:
            print(f'Loaded {sub}')
    if validation:
        return epochs, losses, validations
    return epochs, losses


def handle_multi_index_format(temp_df: pd.DataFrame,
                              mask_type: str = 'events',
                              show_reset_index: bool = False,
                              **format_kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Deals with duplication of event names when printing Multi-indexed Dataframes (or potentially Series).

    Updated and adapted from answer at
    https://stackoverflow.com/questions/75575084/transform-a-pandas-multiindex-to-a-single-index-using-indention

    Parameters
    ----------
    temp_df : Dataframe to use for format. SHOULD BE AN OBJECT MEANT FOR FORMATTING ONLY, copied from original.
    mask_type : Whether to mask repeating events or parameters.
    show_reset_index : Whether to keep or discard default indexes.
    format_kwargs : kwargs to pass to format function (to_markdown, for instance). Either returns them unchanged or
        with index value overridden.

    Returns
    -------
    both DataFrame like and format kwargs

    """
    # we reset the index as mentioned above
    temp_df.reset_index(inplace=True)

    # then find all the duplicates in the zeroth level
    mask = temp_df.events.duplicated().values
    print(mask)

    # and we remove the duplicates
    temp_df.loc[mask, ('events',)] = ''
    if not mask.any():
        mask = temp_df.parameters.duplicated().values
        print(mask)
        temp_df.loc[mask, ('parameters',)] = ''

    if not show_reset_index:
        # Override index specification to avoid showing index 0,1,2...
        format_kwargs['index'] = False

    return temp_df, format_kwargs
