import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import numpy as np
import pandas as pd

import torch

from .common_utils import check_format, get_missing_args
from .train_utils import TrainSet

'''
In this file the raw dataset (list of dictionaries) is transformed
into a dataset compatible with PyTorch [[image0, labels0], ...]

Here regression parameters are also chosen 
'''

no_jargon = {
    'image': 'image',
    'R': 'R',
    'G': 'G',
    'B': 'B',

    'param_pool': None,
    'unit_dict': None,
    'alias_dict': None,

    'default_title_maker': lambda data: 'RGB image'
}


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
        image_arr = np.dstack((image[jargon['R']],
                               image[jargon['G']],
                               image[jargon['B']]))
    else:
        image_arr = np.dstack((data[0], data[1], data[2]))
    return image_arr


# Introduce more redefinition functions if needed

# Load Raw_dataset.pt

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

    for data in dataset:
        params_dict = data['parameters']
        image = data[jargon['image']]
        image_arr = np.array((image[jargon['R']],  # Different shape from make_image
                              image[jargon['G']],
                              image[jargon['B']]))

        new_params_dict = {}
        labels = []  # same info as new_params_dict but in an ordered container
        for param in params_list:
            if param in params_dict:
                new_params_dict[param] = params_dict[param]
            else:  # if not among the base params compute parameter from them
                new_params_dict[param] = calc_parameter(param, param_pool, params_dict)
            labels.append(new_params_dict[param])

        image_list.append(image_arr)
        label_list.append(np.array(labels))
        name_list.append(data['id'])

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

    for inj in dataset:
        params_dict = inj['parameters']

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
    try:
        alias = jargon['alias_dict'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or alias not yet implemented')
        alias = 'unknown alias'
    return alias


def get_param_units(parameter, jargon: dict = None):
    """
    Returns units of given parameter. Used for plotting.
    """
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly convert parameters a jargon["unit_dict"] is required')
    try:
        unit = jargon['unit_dict'][parameter]
    except KeyError:
        print(f'Parameter "{parameter}" misspelled or unit not yet implemented')
        unit = 'unknown unit'
    return unit


def plot_hist(dataset, params_list, fig=None, figsize=None,
              plot_layout=(1, 1, 1), jargon: dict = None,
              *hist_args, **hist_kwargs,):
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
    data = extract_parameters(dataset, params_list)

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
    ax.set_title(f'{names} histogram')
    return fig


def plot_hists(dataset, param_array: np.ndarray, fig=None, figsize=None,
               jargon: dict = None, *hist_args, **hist_kwargs):
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
        fig = plot_hist(dataset, flat_array[i], fig=fig, figsize=figsize, jargon=jargon,
                        plot_layout=(*layout, i + 1), *hist_args, **hist_kwargs)
    plt.tight_layout()
    return fig


def plot_image(data, fig=None, figsize=None, title_maker=None, jargon: dict = None,
               plot_layout=(1, 1, 1), *imshow_args, **imshow_kwargs):
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
    if jargon is None:
        raise RuntimeError('You have no jargon defined: '
                           'To properly plot images a jargon["default_title_maker"] is required')

    if fig is None:
        fig = plt.figure(figsize=figsize)

    image = make_image(data, jargon=jargon)

    ax = fig.add_subplot(*plot_layout)
    ax.imshow(image, *imshow_args, **imshow_kwargs)
    if title_maker is None:
        if isinstance(data, dict):
            ax.set_title(jargon['default_title_maker'](data))
        else:
            pass
    else:
        ax.set_title(title_maker(data))
    return fig


def plot_images(dataset, index_array: np.ndarray, fig=None, figsize=None,
                title_maker=None, jargon: dict = None, *imshow_args, **imshow_kwargs):
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
