import numpy as np
from functools import partial
from dtempest.gw.parameters import *
import dtempest.core.conversion_utils as core

gw_jargon = {
    'image': 'q-transforms',
    'R': 'L1',
    'G': 'H1',
    'B': 'V1',

    'param_pool': redef_dict,
    'unit_dict': unit_dict,
    'alias_dict': alias_dict,

    'default_title_maker': lambda data: f'{data["id"]} Q-Transform image\n(RGB = (L1, H1, V1))'

}

make_image = partial(core.make_image, jargon=gw_jargon)
convert_dataset = partial(core.convert_dataset, jargon=gw_jargon)
extract_parameters = partial(core.extract_parameters, jargon=gw_jargon)
plot_hist = partial(core.plot_hist, jargon=gw_jargon)
plot_hists = partial(core.plot_hists, jargon=gw_jargon)
plot_image = partial(core.plot_image, jargon=gw_jargon)
plot_images = partial(core.plot_images, jargon=gw_jargon)


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



