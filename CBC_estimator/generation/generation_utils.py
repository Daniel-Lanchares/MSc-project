
from pycbc.psd import interpolate, inverse_spectrum_truncation


from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

import numpy as np
from multiprocessing import Value
from skimage.transform import resize


def set_ids(iterable, seed=0):
    valid_id = 1
    for i, injection in enumerate(iterable):
        if injection is not None:
            injection['gen_id'] = i+1
            injection['id'] = f'{seed}.{valid_id:05}'
            valid_id +=1
    return iterable

def cosine(minimum, maximum, size=1, rng=np.random.default_rng()):
    """
    Adapted from bilby.core.prior.analytical.Cosine and its parent class Prior
    Often used for sampling declination
    """
    val = rng.uniform(0, 1, size)
    norm = 1 / (np.sin(maximum) - np.sin(minimum))
    return np.arcsin(val / norm + np.sin(minimum))


def sine(minimum, maximum, size=1, rng=np.random.default_rng()):
    """
    Adapted from bilby.core.prior.analytical.Sine and its parent class Prior
    Often used for sampling some angular parameters
    """
    val = rng.uniform(0, 1, size)
    norm = 1 / (np.cos(minimum) - np.cos(maximum))
    return np.arccos(np.cos(minimum) - val / norm)


def prepare_array(arr):
    """
    Transform q-transform's real part into a 128 x 128 channel of the image
    """
    resol = 128
    # Might make use of the fact that it is still a Spectrogram before casting it
    arr = np.abs(np.flip(arr, axis=1).T/np.max(arr))
    arr = arr[:, 340:900]
    arr = resize(arr, (resol, resol))
    return arr


def image(inject: dict):
    """
    Creates image array from injection's dictionary
    """
    image_arr = np.dstack((inject['q-transforms']['L1'],
                           inject['q-transforms']['H1'],
                           inject['q-transforms']['V1']))
    return image_arr


def validate(inject: dict):
    """
    Decodes validation into string for better printing
    """
    if inject['id'] is not None:
        return 'valid'
    else:
        return 'invalid'


# def write_pt_file(injects: list):
#     """
#     Dumps injections into .pt file for later load from
#     pytorch.
#
#     Right now uses generation ID but will probably end
#     up using its own ID.
#     """
#     for i, inject in enumerate(injects):
#         # Since not all injections are valid,
#         # redefine id as dataset id
#         inject['gen_id'] = inject['id']
#         inject['id'] = i+1
#     filename = f'Dataset/Raw_dataset.pt'
#
#     torch.save(injects, filename)


def get_psd(timeseries, fftlength=4):
    """
    Creates power spectral density of timeseries
    """
    psd_series = timeseries.psd(fftlength)  # Adapted from a workshop, tinker with parameters
    psd_series = interpolate(psd_series, timeseries.delta_f)
    psd_series = inverse_spectrum_truncation(psd_series, int(fftlength * timeseries.sample_rate),
                                             low_frequency_cutoff=15.0)
    return psd_series


def gwpy_filter(timeseries: TimeSeries, detector='L1'):
    """
    Notch filters a gwpy TimeSeries
    """
    # bp = filter_design.bandpass(20, 300, timeseries.sample_rate)  # Donne in whiten()
    if detector != 'V1':
        notches = [filter_design.notch(line, timeseries.sample_rate) for line in (60, 120, 240)]
    else:
        notches = [filter_design.notch(line, timeseries.sample_rate) for line in (50, 100, 150)]
    zpk = filter_design.concatenate_zpks(*notches)
    return timeseries.filter(zpk, filtfilt=True)