from pathlib import Path
from functools import partial

import torch
from pesummary.gw.conversions import convert as pesum_convert

from .config import cbc_jargon
from .parameters import *
import dtempest.core.conversion_utils as core
from dtempest.core.common_utils import check_format
from dtempest.core.train_utils import TrainSet

make_image = partial(core.make_image, jargon=cbc_jargon)
convert_dataset_fast = partial(core.convert_dataset, jargon=cbc_jargon)
extract_parameters = partial(core.extract_parameters, jargon=cbc_jargon)
plot_hist = partial(core.plot_hist, jargon=cbc_jargon)
plot_hists = partial(core.plot_hists, jargon=cbc_jargon)
plot_image = partial(core.plot_image, jargon=cbc_jargon)
plot_images = partial(core.plot_images, jargon=cbc_jargon)


def extract_SNR(dataset, detector_list):
    """
    Extracts an array of SNR peaks of specified
    detectors from a dataset (or its path).
    """

    dataset = core.check_format(dataset)

    SNR_list = []

    for inj in dataset:
        SNR_dict = inj['SNR']

        for ifo in detector_list:
            if ifo in SNR_dict:
                SNR_list.append(SNR_dict[ifo])


def pe_convert_wrapper(target_params, sample_dict):
    converted_dict = pesum_convert(sample_dict)
    labels = []
    for param in target_params:
        try:
            label = converted_dict[param].to_numpy().item()
            labels.append(label)
        except KeyError:
            raise KeyError('The desired parameters could not be achieved with those given.')
    return labels


def convert_dataset_reliable(dataset: str | Path | list | np.ndarray | torch.Tensor,
                             params_list: list | np.ndarray | torch.Tensor,
                             outpath: str | Path = None,
                             name: str = None):
    """
    Unlike its 'fast' counterpart, it outsources its parameter conversion entirely.
    It is reliable and thorough but slow.


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
        name = TrainSet.__name__
    jargon = cbc_jargon
    param_pool = jargon['param_pool']

    image_list, label_list, name_list = [], [], []

    for data in dataset:
        params_dict = data['parameters']
        image = data[jargon['image']]
        image_arr = np.array((image[jargon['R']],  # Different shape from make_image
                              image[jargon['G']],
                              image[jargon['B']]))

        labels = pe_convert_wrapper(params_list, params_dict)

        image_list.append(image_arr)
        label_list.append(np.array(labels))
        name_list.append(data['id'])

    # Tough a roundabout, tensor(array(list)) is the recommended (faster) way
    converted_dataset = TrainSet(data={'images': image_list, 'labels': label_list}, index=name_list, name=name)
    if outpath is not None:
        torch.save(converted_dataset, outpath)
    return converted_dataset


def convert_dataset(*args, method: str = 'default', **kwargs):
    """
    Wrapper class for the two methods of converting between GW parameters. My implementation is faster but less general,
    ideal for low dimensional models. pesummary's implementation is much more thorough and tested, preferable for high
    numbers of parameters.
    The default mode attempts the fast methods and switches is something goes wrong.
    """
    if method == 'default':
        try:
            return convert_dataset_fast(*args, **kwargs)
        except Exception as exception:  # generic exception for now
            print(exception)
            print('Something went wrong in conversion process: Switching to reliable method')
            return convert_dataset_reliable(*args, **kwargs)
    elif method == 'fast':
        return convert_dataset_fast(*args, **kwargs)
    elif method == 'reliable':
        return convert_dataset_reliable(*args, **kwargs)
    else:
        raise TypeError(f'Unsupported method {method}. Suported methods are (default, fast, reliable)')
