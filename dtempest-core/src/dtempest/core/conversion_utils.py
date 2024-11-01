import matplotlib.pyplot as plt
# from pprint import pprint
from pathlib import Path
import numpy as np
# import pandas as pd

import torch

from .config import no_jargon
from .common_utils import check_format, get_missing_args
from .train_utils import TrainSet

'''
In this file the raw dataset (list of dictionaries) is transformed
into a dataset compatible with PyTorch [[image0, labels0], ...]

Here regression parameters are also chosen 
'''


# class CTDataset(Dataset):
#     def __init__(self, filepath):
#         self.x, self.y = torch.load(filepath)
#         self.x = self.x / 255. #Normalize, always good practice
#     def __len__(self): 
#         return self.x.shape[0] #Number of images
#     def __getitem__(self, ix): 
#         return self.x[ix], self.y[ix]

def make_image(data: dict | np.ndarray, jargon: dict = None):
    """
    Creates image array (compatible with plt.imshow()) 
    from injection's dictionary
    """
    if jargon is None:
        jargon = no_jargon
    if isinstance(data, dict):
        image = data[jargon['image']]
        print(image)
        image_arr = np.dstack([image[jargon[channel]]
                               if jargon[channel] in image.keys() else np.zeros_like(list(image.values())[0])
                               for channel in ('R', 'G', 'B')]
                              )
    else:
        image_arr = np.dstack([data[i] for i in range(data.shape[0])])  # Todo: test this option with variable channels
    return image_arr


def make_array(data: dict, jargon: dict = None):
    """
    Creates array apt to be fed to models
    Parameters
    ----------
    data : 
    jargon : 

    Returns
    -------

    """  # TODO: study n-channel cases
    if jargon is None:
        jargon = no_jargon

    image = data[jargon['image']]
    image_arr = np.array((image[jargon['R']],  # Different shape from make_image
                          image[jargon['G']],
                          image[jargon['B']]))
    return image_arr


# Introduce more redefinition functions if needed

# Load Raw_dataset.pt

def data_convert(data: dict, params_list, param_pool, jargon):
    params_dict = data['parameters']
    image = data[jargon['image']]
    channel_list = []
    for val in ('R', 'G', 'B'):  # Sadly, 3-channel images are hard-coded into Pytorch ResNet architectures.
        try:
            channel_list.append(image[jargon[val]])
        except KeyError:
            channel_list.append(np.zeros_like(channel_list[-1]))
    # image_arr = np.array((image[jargon['R']],  # Different shape from make_image
    #                       image[jargon['G']],
    #                       image[jargon['B']]), dtype=np.float32)  # Memory savings. 16 doesn't work with conv2d
    image_arr = np.array(channel_list,
                         dtype=np.float32)

    # new_params_dict = {}
    # labels = []  # same info as new_params_dict but in an ordered container
    # for param in params_list:
    #     if param in params_dict:
    #         new_params_dict[param] = params_dict[param]
    #     else:  # if not among the base params compute parameter from them
    #         new_params_dict[param] = calc_parameter(param, param_pool, params_dict)
    #     labels.append(new_params_dict[param])
    labels = [params_dict[param]
              if param in params_dict else calc_parameter(param, param_pool, params_dict)
              for param in params_list]

    # image_list.append(image_arr)
    # label_list.append(np.array(labels))
    # name_list.append(data['id'])
    return data['id'], image_arr, np.array(labels, dtype=np.float32)  # Memory savings


def convert_dataset(dataset: str | Path | list | np.ndarray | torch.Tensor,
                    params_list: list | np.ndarray | torch.Tensor,
                    outpath: str | Path = None,
                    name: str = None, jargon: dict = None):
    """

    Inputs a raw dataset (list of dictionaries) and outputs a tuple of arrays
    to be used in training.

    Parameters
    ----------
    dataset : string, pathlib.Path or dataset itself.
        Path to the file containing the raw dataset.
    outpath : string or path.
        Path to save the converted dataset in.
        If None file will not be saved
    params_list : list.
        List of parameters to train on.

    Returns
    -------
    converted_dataset : tuple of [data, labels] pairs
        Dataset to actually be trained on.

    """

    dataset = check_format(dataset)
    if name is None:
        if hasattr(dataset, 'name'):
            name = dataset.name
        else:
            name = TrainSet.__name__
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["parameter_pool"] is required')
    param_pool = jargon['param_pool']

    image_list, label_list, name_list = [], [], []

    from multiprocessing import Pool
    from functools import partial
    from tqdm import tqdm

    convert_func = partial(data_convert, params_list=params_list, param_pool=param_pool, jargon=jargon)
    row_dicts = [dict(id=index, **dict(row)) for index, row in dataset.iterrows()]

    with Pool() as pool:  # TODO: configure
        with tqdm(desc='Dataset Conversion', total=len(dataset)) as p_bar:
            for (dat_name, image, label) in pool.imap(convert_func, row_dicts):
                p_bar.update(1)
                name_list.append(dat_name)
                image_list.append(image)
                label_list.append(label)

    # for data in dataset:
    #     params_dict = data['parameters']
    #     image = data[jargon['image']]
    #     image_arr = np.array((image[jargon['R']],  # Different shape from make_image
    #                           image[jargon['G']],
    #                           image[jargon['B']]))
    #
    #     new_params_dict = {}
    #     labels = []  # same info as new_params_dict but in an ordered container
    #     for param in params_list:
    #         if param in params_dict:
    #             new_params_dict[param] = params_dict[param]
    #         else:  # if not among the base params compute parameter from them
    #             new_params_dict[param] = calc_parameter(param, param_pool, params_dict)
    #         labels.append(new_params_dict[param])
    #
    #     image_list.append(image_arr)
    #     label_list.append(np.array(labels))
    #     name_list.append(data['id'])

    converted_dataset = TrainSet(data={'images': image_list, 'labels': label_list}, index=name_list, name=name)
    if outpath is not None:
        converted_dataset.save(outpath)
    return converted_dataset


def calc_parameter(param, pool, params_dict):
    try:
        return pool[param](**params_dict)
    except TypeError as error:
        missing_args = get_missing_args(error)
        missing_dict = {arg: calc_parameter(arg, pool, params_dict) for arg in missing_args}
        return pool[param](**missing_dict, **params_dict)


def extract_parameters(dataset, params_list, jargon: dict = None):
    """
    Extracts an array of specified parameters
    from a dataset (or its path).
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["parameter_pool"] is required')
    param_pool = jargon['param_pool']

    dataset = check_format(dataset)

    label_list = []

    for index, inj in dataset.iterrows():  # TODO: multiprocessing
        params_dict = inj[jargon['parameters']]

        new_params_dict = {}
        labels = []  # same info as new_params_dict but in an ordered container
        for param in params_list:
            if param in params_dict:
                new_params_dict[param] = params_dict[param]
            else:  # if not among the base params compute parameter from them
                new_params_dict[param] = param_pool[param](**params_dict)
            labels.append(new_params_dict[param])

        label_list.append(np.array(labels))
    return np.array(label_list)


def get_param_alias(parameter, jargon: dict = None):
    """
    Returns alias of given parameter. Used for plotting.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["alias_dict"] is required')
    if jargon['labels'] is None:
        return 'unknown unit'
    try:
        label = jargon['labels'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or unit not yet implemented')
        return 'unknown alias'
    split = label.split(' ')
    if len(split) == 1:
        alias = split[0][1:-1]
    elif len(split) == 2:
        alias = split[0][1:]
    else:
        alias = ''  # If this executes label format is not right ($alias [unit]$)
    return r'${}$'.format(alias)


def get_param_units(parameter, jargon: dict = None):
    """
    Returns units of given parameter. Used for plotting.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["unit_dict"] is required')
    if jargon['labels'] is None:
        return 'unknown unit'
    try:
        label = jargon['labels'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or unit not yet implemented')
        return 'unknown unit'

    try:
        unit = label.split(' ')[1][1:-2]
    except IndexError:
        return r'$ø$'
    return r'${}$'.format(unit)


def plot_hist(dataset, params_list, fig=None, figsize=None, title: str = 'default',
              plot_layout=(1, 1, 1), jargon: dict = None,
              *hist_args, **hist_kwargs, ):
    """
    Plots a histogram of a given parameter list on a single subplot

    Parameters
    ----------
    dataset : TYPE
        Raw_dataset.
    params_list : list.
        List of parameters to plot.
    fig : matplotlib.pyplot.figure, optional
        Matplotlib.pyplot figure to be plotted on. Especially useful to paint
        various plots manually. The default is None.
    figsize : tuple, optional.
        'figsize' parameter for fig. The default is None.
    plot_layout : tuple, optional
        Plot layout. Useful mostly to paint
        various plots manually. Only The default is (1,1,1).
    *hist_args : iterable.
        Arguments to be passed to plt.hist().
    **hist_kwargs : dict.
        Keyword arguments to be passed to plt.hist().

    Returns
    -------
    fig : matplotlib.pyplot.figure
        updated figure with the histogram now plotted.

    """

    dataset = check_format(dataset)
    data = extract_parameters(dataset, params_list, jargon)

    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(*plot_layout)
    ax.hist(data, *hist_args, **hist_kwargs)
    # Study multiple label for lists, probably useless though
    ax.set_xlabel(f'{get_param_alias(params_list[0], jargon)} ({get_param_units(params_list[0], jargon)})')
    # Do the same for set_title ?
    names = ''
    for name in params_list:
        names += get_param_alias(name, jargon) + ', '
    names = names[:-2]
    if title in ['default', None]:
        ax.set_title(f'{names} histogram')
    else:
        ax.set_title(title)
    return fig


def plot_hist_ax(dataset,
                 params_list: list | str,
                 ax: plt.Axes,
                 title: str = 'default', jargon: dict = None, *hist_args, **hist_kwargs):
    dataset = check_format(dataset)
    if isinstance(params_list, str):
        params_list = [params_list, ]
    data = extract_parameters(dataset, params_list, jargon)
    ax.hist(data, *hist_args, **hist_kwargs)
    # Study multiple label for lists, probably useless though
    ax.set_xlabel(f'{get_param_alias(params_list[0], jargon)} ({get_param_units(params_list[0], jargon)})')
    # Do the same for set_title ?
    names = ''
    for name in params_list:
        names += get_param_alias(name, jargon) + ', '
    names = names[:-2]
    if title in ['default', None]:
        ax.set_title(f'{names} histogram')
    else:
        ax.set_title(title)


def plot_hists(dataset,
               param_array: np.ndarray,
               fig=None,
               figsize=None,
               title: str = 'default',
               jargon: dict = None,
               *hist_args, **hist_kwargs):
    """
    Plots histograms of the given parameter array on one or more subplots

    Parameters
    ----------
    dataset : TYPE
        Raw_dataset.
    param_array : np.ndarray
        Array of parameters to plot. Dictates figure layout
    fig : matplotlib.pyplot.figure, optional
        Matplotlib.pyplot figure to be plotted on. Especially useful to paint
        various plots manually. The default is None.
    figsize : tuple, optional
        'figsize' parameter for fig. The default is None.
    *hist_args : iterable
        Arguments to be passed to plt.hist().
    **hist_kwargs : dict
        Keyword arguments to be passed to plt.hist().

    Returns
    -------
    fig : matplotlib.pyplot.figure
        updated figure with the histograms now plotted.

    """
    # Study way of having different args and kwargs for each hist

    if fig is None:
        fig = plt.figure(figsize=figsize)

    layout = param_array.shape
    flat_array = param_array.flatten()
    for i in range(len(flat_array)):
        fig = plot_hist(dataset, [flat_array[i], ], fig=fig, figsize=figsize, title=title, jargon=jargon,
                        plot_layout=(*layout, i + 1), *hist_args, **hist_kwargs)
    plt.tight_layout()
    return fig


def plot_image(data, fig=None, figsize=None, title_maker=None, jargon: dict = None,
               plot_layout=(1, 1, 1), title_kwargs=None, *imshow_args, **imshow_kwargs):
    """
        Plots a histogram of a given parameter list on a single subplot

        Parameters
        ----------
        data : dict
            element of the Raw_dataset.
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional
            'figsize' parameter for fig. The default is None.
        title_maker : callable, optional
            funtion to return image_title base on injection. The default is None.
        plot_layout : tuple, optional
            Plot layout. Useful mostly to paint
            various plots manually. Only The default is (1,1,1).
        *imshow_args : iterable
            Arguments to be passed to plt.imshow().
        **imshow_kwargs : dict
            Keyword arguments to be passed to plt.imshow().

        Returns
        -------
        fig : matplotlib.pyplot.figure
            updated figure with the image now plotted.

        """
    if title_kwargs is None:
        title_kwargs = {}
    if jargon is None:
        jargon = no_jargon

    if fig is None:
        fig = plt.figure(figsize=figsize)

    image = make_image(data, jargon=jargon)

    ax = fig.add_subplot(*plot_layout)
    ax.imshow(image, *imshow_args, **imshow_kwargs)
    if title_maker is None:
        if isinstance(data, dict):
            ax.set_title(jargon['default_title_maker'](data), **title_kwargs)
        else:
            pass
    else:
        ax.set_title(title_maker(data), **title_kwargs)
    return fig


def plot_images(dataset,
                index_array: np.ndarray,
                fig=None,
                figsize=None,
                title_maker=None,
                jargon: dict = None,
                *imshow_args, **imshow_kwargs):
    """
        Plots histograms of the given parameter array on one or more subplots

        Parameters
        ----------
        dataset : TYPE
            Raw_dataset.
        index_array : np.ndarray
            Array of indexes of the dataset to plot. Dictates figure layout
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional
            'figsize' parameter for fig. The default is None.
        title_maker : callable, optional
            funtion to return image_title base on injection. The default is None.
        *imshow_args : iterable
            Arguments to be passed to plt.imshow().
        **imshow_kwargs : dict
            Keyword arguments to be passed to plt.imshow().

        Returns
        -------
        fig : matplotlib.pyplot.figure
            updated figure with the images now plotted.

        """
    # Study way of having different args and kwargs for each hist

    if fig is None:
        fig = plt.figure(figsize=figsize)

    dataset = check_format(dataset)
    layout = index_array.shape
    flat_array = index_array.flatten()
    for i, indx in enumerate(flat_array):
        fig = plot_image(dataset[indx], fig=fig, figsize=figsize, title_maker=title_maker, jargon=jargon,
                         plot_layout=(*layout, i + 1), *imshow_args, **imshow_kwargs)
    plt.tight_layout()
    return fig
