import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from cycler import cycler
import matplotlib.pyplot as plt
from collections import OrderedDict


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


class Pallete:

    def __init__(self, n: int = 10, cmap: str = 'jet'):
        self.cycler = colour_cycler(n, cmap)

    def colour(self):
        return next(self.cycler())

    def colours(self):
        return self.cycler

    # def merge(self, other):
    #     ex = []
    #     for b, o in blues, oranges:
    #         ex.append(b)
    #         ex.append(o)
    #     cyc = cycle


def colour_cycler(n: int = 10, cmap: str = 'jet'):
    return cycler('color', [plt.get_cmap(cmap)(1. * i / n) for i in range(1, n)])


def change_legend_loc(artist, loc: str | int, pos: int = 0):
    """

    Parameters
    ----------
    artist : holder of the legend
    loc : new position
    pos : if there al multiple legends, specify position on list

    Unlike redraw_legend, it is non-destructive
    -------

    """
    if isinstance(artist, plt.Axes):
        legend = artist.get_legend()
    elif isinstance(artist, plt.Figure):
        legend = artist.legends[pos]  # A figure can have multiple legends
    else:
        raise NotImplementedError(f"Change of legend location is not implemented for objects of type '{type(artist)}'")

    if isinstance(loc, str):
        from matplotlib.offsetbox import AnchoredOffsetbox
        try:
            legend._loc = AnchoredOffsetbox.codes[loc]
        except KeyError as e:
            raise KeyError(f"Location '{loc}' not valid").with_traceback(e.__traceback__)
    elif isinstance(loc, int):
        legend._loc = loc
    else:
        raise ValueError(f"Paramemer loc can only be of class 'int' or 'str', not '{type(loc)}'")


def redraw_legend(artist, *args, pos: int = 0, **kwargs):
    from matplotlib.legend import Legend
    # from matplotlib.font_manager import FontProperties
    # from matplotlib.legend import legend_handler
    if isinstance(artist, plt.Axes):
        l: Legend = artist.get_legend()
    elif isinstance(artist, plt.Figure):
        l: Legend = artist.legends[pos]  # A figure can have multiple legends
    else:
        raise NotImplementedError(f"Change of legend location is not implemented for objects of type '{type(artist)}'")
    # l.prop = FontProperties(size=fontsize)
    # l._fontsize = l.prop.get_size_in_points()
    # for text in l.texts:
    #     text.set_fontsize(fontsize)
    # texts = [text.get_text() for text in l.texts]
    handles = l.legend_handles

    linewidth = kwargs.pop('linewidth', None)
    # Do a more robust implementation for all handler properties if needed
    if linewidth is not None:
        for handle in handles:
            handle.set_linewidth(linewidth)

    l.remove()
    artist.legend(handles=handles, *args, **kwargs)


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


def rawset_setup(data, obj):
    if 'id' in data:
        obj[data['id']] = data
    else:
        obj[len(obj)] = data  # Untested


class RawSet:
    """
    Iterable of OrderedDicts containing fully described events (injections in GW)
    Could be a subclass of list...
    """

    def __init__(self, rawdata, name: str = None, metadata: list[dict] = None, **fdict_kwargs):
        # self._df = pd.Series(*args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name
        self.metadata = metadata

        if isinstance(rawdata, pd.DataFrame):
            self._df = rawdata
        else:
            # from multiprocessing import Pool, Manager
            # from functools import partial
            # from tqdm import tqdm
            #
            # shared_dict = Manager().dict()
            # setup_func = partial(rawset_setup, obj=shared_dict)
            #
            # with Pool(**pool_kwargs) as pool:
            #     with tqdm(desc='Rawset Creation', total=len(rawdata)) as p_bar:
            #         for _ in pool.imap(setup_func, rawdata):
            #             p_bar.update(1)

            shared_dict = {data['id']: data for data in rawdata}

            self._df = pd.DataFrame.from_dict(data=shared_dict, orient='index', **fdict_kwargs).drop('id', axis=1)
            # print(self._df)

        # for data in rawdata:
        #     if 'id' in data:
        #         self[data['id']] = data
        #     else:
        #         self[len(self._df)] = data  # Untested

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        # if isinstance(self._df[item], pd.Series):  # Need to study whether I will actually use this
        #     return type(self)(rawdata=self._df[item], name=self.name)
        if isinstance(self._df[item], pd.DataFrame):
            return type(self)(rawdata=self._df[item], name=self.name)
        return self._df[item]  # Changed to avoid warnings, might break something (self._df[item])

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return self._df.__repr__()

    def change_parameter_name(self, old_key, to):  # to: New key
        # for index, row in self._df.iterrows():
        #     # params = row['parameters']
        #     # data['parameters']['luminosity_distance'] = data['parameters'].pop('d_L')
        #     row['parameters'] = OrderedDict(
        #         [(to, v) if k == old_key else (k, v) for k, v in row['parameters'].items()])
        self._df.loc[:, 'parameters'] = self._df.loc[:, 'parameters'].apply(
            lambda x: OrderedDict([(to, v) if k == old_key else (k, v) for k, v in x.items()]))

    def __len__(self):
        return self._df.__len__()


def seeds2names(seeds, zero_pad: int = 3):
    if not hasattr(seeds, '__iter__'):
        seeds = [seeds, ]
    return [f'Raw_Dataset_{seed:0{zero_pad}}.pt' for seed in seeds]


def load_rawset(dire):
    return torch.load(dire)


def load_rawsets(directory, names: list[str], verbose: bool = True):
    if not hasattr(names, '__iter__'):
        names = [names, ]
    # directories = [Path(directory) / name for name in names]
    dataset = []
    metadata = []
    seeds = {}
    for i, name in enumerate(names):
        loaded = load_rawset(directory / name)
        if isinstance(loaded, InjectionList):
            seed = loaded.metadata.pop('seed')
            loaded.metadata['seed'] = None
            cache_check = np.array([loaded.metadata == data for data in metadata])
            if i == 0:
                metadata.append(loaded.metadata)
                seeds[0] = [seed, ]
            elif np.any(cache_check):
                seeds[np.where(cache_check)[0].item()].append(seed)
            else:
                metadata.append(loaded.metadata)
                seeds[len(metadata) - 1] = [seed, ]

        dataset = np.concatenate((dataset, loaded))
        if verbose:
            print(f'Loaded {name}')
    for num, seed_list in seeds.items():
        metadata[num]['seed'] = seed_list

    return RawSet(dataset, name='+'.join(names), metadata=metadata)


def load_rawsets_pool(directory, names: str | list[str], verbose: bool = True, **pool_kwargs):
    if not hasattr(names, '__iter__'):
        names = [names, ]
    directories = [Path(directory) / name for name in names]
    dataset = []
    metadata = []
    seeds = {}
    with Pool(**pool_kwargs) as pool:
        for i, loaded in enumerate(pool.imap(load_rawset, directories)):
            if isinstance(loaded, InjectionList):
                seed = loaded.metadata.pop('seed')
                loaded.metadata['seed'] = None
                cache_check = np.array([loaded.metadata == data for data in metadata])
                if i == 0:
                    metadata.append(loaded.metadata)
                    seeds[0] = [seed, ]
                elif np.any(cache_check):
                    seeds[np.where(cache_check)[0].item()].append(seed)
                else:
                    metadata.append(loaded.metadata)
                    seeds[len(metadata) - 1] = [seed, ]

            dataset = np.concatenate((dataset, loaded))
            if verbose:
                print(f'Loaded {names[i]}')
        for num, seed_list in seeds.items():
            metadata[num]['seed'] = seed_list

    return RawSet(dataset, name='+'.join(names), metadata=metadata)


class InjectionList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metadata = None

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict = None):
        self._metadata = value


def load_losses(directory: str | Path,
                model: str = None,
                stages: int | list[int] = None,
                validation: bool = False,
                verbose: bool = True) -> tuple[dict, dict] | tuple[dict, dict, dict, dict]:
    subdirs = [f.name for f in os.scandir(directory) if f.is_dir()]
    if stages is None:
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs}
    else:
        if not hasattr(stages, '__iter__'):
            stages = [stages, ]
        chosen_subs = {int(subdir.split('_')[-1]): subdir for subdir in subdirs  # add '#' to ignore directory
                       if int(subdir.split('_')[-1]) in stages and subdir.split('_')[0] != '#'}

    if model is not None:
        chosen_subs = {stage: subdir for stage, subdir in chosen_subs.items() if subdir.split('_')[-3] == model}

    # Sort the stages to avoid issues down the line
    chosen_subs = {stage: chosen_subs[stage] for stage in sorted(chosen_subs.keys())}

    epochs = {}
    losses = {}
    vali_epochs = {}
    validations = {}
    last_epoch = 0  # Should never be needed, but suppresses static warning
    for i, (stage, sub) in enumerate(chosen_subs.items()):
        epoch, loss = torch.load(Path(directory) / sub / 'loss_data.pt')

        if len(epoch.shape) == 1:  # To deal with legacy loss format
            nepochs = int(epoch[-1]) + 1
            epoch = epoch.reshape((nepochs, -1))
            loss = loss.reshape((nepochs, -1))

        if i == 0:
            epochs[stage] = epoch

        else:
            epochs[stage] = epoch + last_epoch * np.ones_like(epoch)

        losses[stage] = loss
        if validation:
            try:
                vali_epoch, valid = torch.load(Path(directory) / sub / 'validation_data.pt')
                if len(vali_epoch.shape) == 1:
                    nepochs = int(vali_epoch[-1]) + 1
                    vali_epoch = vali_epoch.reshape((nepochs, -1))
                    valid = valid.reshape((nepochs, -1))
                if i == 0:
                    vali_epochs[stage] = vali_epoch
                else:
                    vali_epochs[stage] = vali_epoch + last_epoch * np.ones_like(vali_epoch)
                validations[stage] = valid
            except FileNotFoundError as exc:
                print(f'Validation not found in {sub}')
                print(exc)
        if verbose:
            print(f'Loaded {sub}')
        last_epoch = epochs[stage].flatten()[-1]
    if validation:
        return epochs, losses, vali_epochs, validations
    return epochs, losses


def handle_multi_index_format(temp_df: pd.DataFrame,
                              # mask_type: str = 'events',
                              show_reset_index: bool = False,
                              **format_kwargs) -> tuple[pd.DataFrame, dict]:
    """
    Deals with duplication of event names when printing Multi-indexed Dataframes (or potentially Series).

    Updated and adapted from answer at
    https://stackoverflow.com/questions/75575084/transform-a-pandas-multiindex-to-a-single-index-using-indention

    Parameters
    ----------
    temp_df : Dataframe to use for format. SHOULD BE AN OBJECT MEANT FOR FORMATTING ONLY, copied from original.
    # mask_type : Whether to mask repeating events or parameters.
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


def merge_headers(string_table: str):
    """
    Final stop of latex table custom formatting

    Parameters
    ----------
    string_table : latex table as a continuous string

    Returns The string with merged headers (index name should now show next to column names)
    -------

    """
    rows = string_table.split(r' \\')
    problem_name = rows.pop(1).split('&')[0]
    rows[0] = rows[0].replace(r'\toprule', r'\toprule' + problem_name)

    return r' \\'.join(rows)
