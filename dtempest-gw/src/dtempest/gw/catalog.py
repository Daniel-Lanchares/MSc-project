"""This module adds catalog related functionality to test models on real data"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from pycbc.catalog import Catalog as pycbc_Catalog
from pycbc.catalog import Merger as pycbc_Merger
from pycbc.catalog import _aliases
from pycbc.catalog.catalog import get_source

from dtempest.core.conversion_utils import make_image, make_array
from dtempest.gw.conversion import cbc_jargon
from dtempest.gw.generation.generation_utils import get_psd
from dtempest.gw.generation.parallel import process_strain, whiten
from dtempest.gw.generation.parallel import default_config as default_gen_config

full_names = {
    'GW190412': 'GW190412_053044',
    'GW190425': 'GW190425_081805',
    'GW190521': 'GW190521_030229',
    'GW190814': 'GW190814_211039'
}


class Merger(pycbc_Merger, dict):
    def __init__(self,
                 name: str,
                 source: str | dict = 'gwtc-1',
                 image_window: tuple = None,
                 img_res: tuple[int, int] = (128, 128),
                 frange: tuple[float, float] = (20.0, 300.0),
                 old_pipe: bool = False):
        """ Return the information of a merger

        Parameters
        ----------
        name: str
            The name (GW prefixed date) of the merger event.
        """
        self.available_parameters = []

        if isinstance(source, dict):
            self.source = source
        else:
            self.source = get_source(source)

        try:
            self.data = self.source[name]
        except KeyError:
            # Try common name
            for mname in self.source:
                cname = self.source[mname]['commonName']
                if cname == name:
                    name = mname
                    self.data = self.source[name]
                    break
            else:
                raise ValueError('Did not find merger matching'
                                 ' name: {}'.format(name))

        # Set some basic params from the dataset
        for key in self.data:
            setattr(self, '_raw_' + key, self.data[key])
            if key not in ['commonName', 'version', 'catalog.shortName', 'reference', 'jsonurl', 'strain']:
                if key[-5:] not in ['lower', 'upper', '_unit'] and self.data[key] is not None:
                    self.available_parameters.append(key)

        for key in _aliases:
            setattr(self, key, self.data[_aliases[key]])

        self.detectors = list({d['detector'] for d in self.data['strain']})
        self.common_name = self.data['commonName']
        self.time = self.data['GPS']
        self.frame = 'source'
        self.img_res = img_res
        self.frange = frange
        self.old_pipe = old_pipe

        if image_window is None:
            self.image_window = default_gen_config['q_interval']
            print(f'Warning: No image_window specified. Resorting to default: {self.image_window}')
        else:
            self.image_window = image_window
        # print(name)
        self.qtransforms = self.process_strains()
        for ifo in ('L1', 'H1', 'V1'):
            if ifo not in self.detectors:
                self.qtransforms[ifo] = np.zeros_like(self.qtransforms[self.detectors[0]])

    def __getitem__(self, key):
        if key == 'q-transforms' or key == 'image':
            return self.qtransforms
        elif key == 'parameters' or key == 'labels':
            return {parameter: self.data[parameter] for parameter in self.available_parameters}
        elif key == 'id':
            return self.common_name
        else:
            raise KeyError(f'{type(self)} does not have the attribute "{key}"')

    def __repr__(self):
        items = list(self['parameters'].items())
        rep = "{'" + str(items[0][0]) + "': " + str(items[0][1]) + " ... }"
        return rep

    def process_strains(self):
        from .generation.images import ifo_q_transform
        from gwpy.timeseries import TimeSeries
        channels = {}
        for ifo in self.detectors:
            ts = self.strain(ifo)
            if self.old_pipe:
                q_window = (self.time + self.image_window[0], self.time + self.image_window[1])
                channels[ifo] = process_strain(ts._return(np.nan_to_num(ts)), ifo, q_window)
            else:
                ts = TimeSeries.from_pycbc(whiten(ts, get_psd(ts)))
                ts = ts.crop(self.time - 1, self.time + 1)
                channels[ifo] = ifo_q_transform(ts.value,
                                                resol=self.img_res,
                                                duration=ts.duration,
                                                sampling_frequency=ts.sample_rate,
                                                outseg=self.image_window,
                                                frange=self.frange)
        return channels

    def make_image(self):
        return make_image(self, cbc_jargon)

    def make_array(self):
        return make_array(self, cbc_jargon)

    def imshow(self, ax=None, *args, **kwargs):
        if ax is None:
            return plt.imshow(make_image(self, cbc_jargon), *args, **kwargs)
        return ax.imshow(make_image(self, cbc_jargon), *args, **kwargs)


# default_config = {
#     'img_res': (128, 128)
# }


class Catalog(pycbc_Catalog):
    def __init__(self,
                 source: str = 'gwtc-1',
                 # config: dict = None,
                 **merger_kwargs):
        # if config is None:
        #     config = {}

        self.source = source

        # self.config = deepcopy(default_config)
        # for attr, val in config.items():
        #     if attr not in self.config.keys():
        #         raise KeyError(f'Specified key ({attr}) misspelled or not implemented')
        #     self.config[attr] = val

        self.data = get_source(source=self.source)
        self.mergers = {name: Merger(name,
                                     source=source,
                                     **merger_kwargs) for name in self.data.keys() if name != 'GW190720_000836-v2'}  # TODO: fix?
        self.names = self.mergers.keys()
        self.common_names = [self.mergers[m].common_name for m in self.mergers]

    def __delitem__(self, key):
        try:
            del self.mergers[key]
        except KeyError:
            # Try common name
            for m in self.mergers:
                if key == self.mergers[m].common_name:
                    break
            else:
                raise ValueError('Did not find merger matching'
                                 ' name: {}'.format(key))
            del self.mergers[m]


# c = Catalog('gwtc-1')
# print(c['GW150914'].data)
# m = Merger('GW150914', 'gwtc-1')

# for name, merger in sorted(Catalog('gwtc-1').mergers.items()):
#     merger.imshow()
#     plt.title(name)
#     plt.show()
# Sort-of-DONE: Add add_parameter / remove_parameter routines to both merger and catalog (does it with PESummary)


'''
From here down an exploration of sample download to compare actual distributions an obtain more parameters 
(much later down the line)
'''
# DONE: build a table of medians + 90% credible intervals of he GWTC-3 catalog by downloading all 90+ files
# "IGWN-GWTC3p0-v1-FULLNAME_PEDataRelease_mixed_cosmo.h5"
# "IGWN-GWTC2p1-v2-FULLNAME_PEDataRelease_mixed_cosmo.h5" # Up to 190930

# import h5py
# from pathlib import Path
# base_path = Path('/home/daniel/Descargas')
# file_name = "IGWN-GWTC3p0-v1-GW200224_222234_PEDataRelease_mixed_cosmo.h5"
#
# with h5py.File(base_path/file_name, "r") as f:
#     print("H5 data sets:")
#     print(list(f))
#
# from pesummary.io import read
# # Reading with PESummary directly
# data = read(base_path/file_name)
# print("Run labels:")
# print(data.labels)
#
# import matplotlib.pyplot as plt
#
# samples_dict = data.samples_dict
# posterior_samples = samples_dict["C01:Mixed"]
# parameters = posterior_samples.parameters
# print(parameters)
# fig = posterior_samples.plot("chirp_mass_source", type="hist", kde=True)
# plt.show()
