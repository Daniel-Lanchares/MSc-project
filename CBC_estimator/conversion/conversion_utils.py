import matplotlib.pyplot as plt
from pprint import pprint
from pathlib import Path
import numpy as np

import torch

# from torch.utils.data import Dataset, DataLoader

'''
In this file the raw dataset (list of dictionaries) is transformed
into a dataset compatible with PyTorch [[image0, labels0], ...]

Here regression parameters are also chosen 
'''
# ['dicts',]
debug = ['dicts']


# class CTDataset(Dataset):
#     def __init__(self, filepath):
#         self.x, self.y = torch.load(filepath)
#         self.x = self.x / 255. #Normalize, always good practice
#     def __len__(self): 
#         return self.x.shape[0] #Number of images
#     def __getitem__(self, ix): 
#         return self.x[ix], self.y[ix]

def make_image(inject: dict | np.ndarray):
    """
    Creates image array (compatible with plt.imshow()) 
    from injection's dictionary
    """
    if isinstance(inject, dict):
        image_arr = np.dstack((inject['q-transforms']['L1'],
                               inject['q-transforms']['H1'],
                               inject['q-transforms']['V1']))
    else:
        image_arr = np.dstack((inject[0], inject[1], inject[2]))
    return image_arr


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


# Parameter redefinitions.
# kwargs is used because we will input all base parameters to all functions

def chirp_mass(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given 15 base parameters returns chirp mass.
    """
    return pow(mass_1 * mass_2, 3 / 5) / pow(mass_1 + mass_2, 1 / 5)


def mass_ratio(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given 15 base parameters returns mass ratio.
    """
    return np.minimum(mass_1, mass_2) / np.maximum(mass_1, mass_2)


def symmetric_mass_ratio(mass_1, mass_2, **kwargs):
    """
    Conversion function: Given 15 base parameters returns symmetric mass ratio.
    """
    q = mass_ratio(mass_1, mass_2)
    return q / (1 + q ** 2)


def chi_1(a_1, tilt_1=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns parallel component of
    unit-less spin.

    If tilt_1 isn't provided assumes aligned spin (might introduce anti-align
    possibility with 50/50 chance of 0 or np.pi).
    """
    return a_1 * np.cos(tilt_1)


def chi_2(a_2, tilt_2=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns parallel component of
    unit-less spin.

    If tilt_2 isn't provided assumes aligned spin (might introduce anti-align
    possibility with 50/50 chance of 0 or np.pi).
    """
    return a_2 * np.cos(tilt_2)


def chi_eff(mass_1, mass_2, a_1, a_2, tilt_1=0, tilt_2=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns effective spin.
    """
    chi1 = chi_1(a_1, tilt_1)
    chi2 = chi_2(a_2, tilt_2)
    return (mass_1 * chi1 + mass_2 * chi2) / (mass_1 + mass_2)


def chi_1_in_plane(a_1, tilt_1=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns the perpendicular
    component of unit-less spin.
    """
    return np.abs(a_1 * np.sin(tilt_1))


def chi_2_in_plane(a_2, tilt_2=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns the perpendicular
    component of unit-less spin.
    """
    return np.abs(a_2 * np.sin(tilt_2))


def chi_p(mass_1, mass_2, a_1, a_2, tilt_1=0, tilt_2=0, **kwargs):
    """
    Conversion function: Given 15 base parameters returns precession spin.
    """
    q = mass_ratio(mass_1, mass_2)
    chi1_p = chi_1_in_plane(a_1, tilt_1)
    chi2_p = chi_2_in_plane(a_2, tilt_2)
    return np.maximum(chi1_p, q * (3 * q + 4) / (4 * q + 3) * chi2_p)


redef_dict = {  # MANY MISSING (redshift) #TODO
    'chirp_mass': chirp_mass,
    'mass_ratio': mass_ratio,
    'symmetric_mass_ratio': symmetric_mass_ratio,
    'chi_eff': chi_eff,
    'chi_p': chi_p,
    'chi_1': chi_1,
    'chi_2': chi_2,
    'chi_1_in_plane': chi_1_in_plane,
    'chi_2_in_plane': chi_2_in_plane,
}

unit_dict = {  # MANY MISSING #TODO
    'mass_1': r'$M_{\odot}$',
    'mass_2': r'$M_{\odot}$',
    'chirp_mass': r'$M_{\odot}$',
    'mass_ratio': r'$ø$',
    'symetric_mass_ratio': r'$ø$',
    'NAP': r'$ø$',
    'chi_eff': r'$ø$',
    'chi_p': r'$ø$',
    'd_L': r'$Mpc$'
}

alias_dict = {  # MANY MISSING #TODO
    'mass_1': r'$m_1$',
    'mass_2': r'$m_2$',
    'chirp_mass': r'$\mathcal{M}$',
    'mass_ratio': r'$q$',
    'symetric_mass_ratio': r'$\eta$',
    'NAP': 'Network Antenna Pattern',
    'chi_eff': r'$\chi_{eff}$',
    'chi_p': r'$\chi_{p}$',
    'd_L': r'$d_L$'
}


# Introduce more redefinition functions if needed

# Load Raw_dataset.pt

def convert_dataset(dataset, params_list, outpath=None, debug=None):
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
    debug : list, optional.
        debuging options. The default is [].

    Returns
    -------
    converted_dataset : tuple of [data, labels] pairs
        Dataset to actually be trained on.

    """

    if debug is None:
        debug = []
    dataset = check_format(dataset)

    image_list, label_list = [], []

    for inj in dataset:
        params_dict = inj['parameters']
        image_arr = np.array((inj['q-transforms']['L1'],
                              inj['q-transforms']['H1'],
                              inj['q-transforms']['V1']))  # image(inj)

        new_params_dict = {}
        labels = []  # same info as new_params_dict but in an ordered container
        for param in params_list:
            if param in params_dict:
                new_params_dict[param] = params_dict[param]
            else:  # if not among the base params compute parameter from them
                new_params_dict[param] = redef_dict[param](**params_dict)
            labels.append(new_params_dict[param])

        image_list.append(image_arr)
        label_list.append(np.array(labels))

        if 'dicts' in debug:
            inj_id = inj['id']
            print(f'Injection No.{inj_id}')
            print('Full parameters')
            pprint(params_dict)
            print()
            print('Selected parameters')
            pprint(new_params_dict)
            print('\n' * 2)

    # Tough a roundabout, tensor(array(list)) is the recomended (faster) way
    converted_dataset = (torch.tensor(np.array(image_list)),
                         torch.tensor(np.array(label_list)))
    if outpath is not None:
        torch.save(converted_dataset, outpath)
    return converted_dataset


def extract_parameters(dataset, params_list):
    """
    Extracts an array of specified parameters
    from a dataset (or its path).
    """

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
                new_params_dict[param] = redef_dict[param](**params_dict)
            labels.append(new_params_dict[param])

        label_list.append(np.array(labels))
    return np.array(label_list)


def extract_SNR(dataset, detector_list):
    """
    Extracts an array of SNR peaks of specified
    detectors from a dataset (or its path).
    """

    dataset = check_format(dataset)

    SNR_list = []

    for inj in dataset:
        SNR_dict = inj['SNR']

        for ifo in detector_list:
            if ifo in SNR_dict:
                SNR_list.append(SNR_dict[ifo])


def get_param_alias(parameter):
    """
    Returns alias of given parameter. Used for plotting.
    """
    try:
        alias = alias_dict[parameter]
    except KeyError:
        print('Parameter misspelled or alias not yet implemented')
        alias = 'unknown alias'
    return alias


def get_param_units(parameter):
    """
    Returns units of given parameter. Used for plotting.
    """
    try:
        unit = unit_dict[parameter]
    except KeyError:
        print('Parameter misspelled or unit not yet implemented')
        unit = 'unknown unit'
    return unit


def plot_hist(dataset, params_list, fig=None, figsize=None,
              plot_layout=(1, 1, 1), *hist_args, **hist_kwargs):
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
    ax.set_xlabel(f'{get_param_alias(params_list[0])} ({get_param_units(params_list[0])})')
    # Do the same for set_title ?
    names = ''
    for name in params_list:
        names += get_param_alias(name) + ', '
    names = names[:-2]
    ax.set_title(f'{names} histogram')
    return fig


def plot_hists(dataset, param_array: np.ndarray, fig=None, figsize=None,
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
        fig = plot_hist(dataset, flat_array[i], fig=fig, figsize=figsize,
                        plot_layout=(*layout, i + 1), *hist_args, **hist_kwargs)
    plt.tight_layout()
    return fig


def plot_image(injection, fig=None, figsize=None, title_maker=None,
               plot_layout=(1, 1, 1), *imshow_args, **imshow_kwargs):
    """
        Plots a histogram of a given parameter list on a single subplot

        Parameters
        ----------
        injection : dict
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

    if fig is None:
        fig = plt.figure(figsize=figsize)

    image = make_image(injection)

    ax = fig.add_subplot(*plot_layout)
    ax.imshow(image, *imshow_args, **imshow_kwargs)
    if title_maker is None:
        ax.set_title(f'{injection["id"]} Q-Transform image\n(RGB = (L1, H1, V1))')
    else:
        ax.set_title(title_maker(injection))
    return fig


def plot_images(dataset, index_array: np.ndarray, fig=None, figsize=None,
                title_maker=None, *imshow_args, **imshow_kwargs):
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
        fig = plot_image(dataset[indx], fig=fig, figsize=figsize, title_maker=title_maker,
                         plot_layout=(*layout, i + 1), *imshow_args, **imshow_kwargs)
    plt.tight_layout()
    return fig
