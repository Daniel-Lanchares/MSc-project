# import numpy as np
# import bilby
# import torch
# from pathlib import Path
# import matplotlib.pyplot as plt
from pathlib import Path
from pprint import pprint

from gwpy.timeseries import TimeSeries

# from dtempest.gw.conversion import make_image
# from dtempest.gw.generation.timeseries import get_data
# from dtempest.gw.generation.images import dataset_q_transform
# from dtempest.gw.generation.wavelet import Morlet, CWT

import warnings

# bilby.utils.logging.disable(level=50)
# warnings.filterwarnings("ignore")  # Ignore the annoying default warnings
#
# files_path = Path('/media/daniel/easystore/Daniel/MSc-files/Raw Datasets/new pipe')
# zero_pad = 3
#
# joblib_kwargs = {
#     'n_jobs': 12,
#     'backend': 'multiprocessing'
# }
# # %%
# duration = 2.0  # seconds
# sampling_frequency = 1024  # Hz  DON'T KNOW WHY, BUT DO NOT MESS WITH IT!
# min_freq = 20.0  # Hz
#
# ifolist = ['L1', 'H1', 'V1']
# # asds = []
# # noiselist = []
# # for ifo in ifolist:
# #     Read noise from file, notch_filter and resample to 2048 Hz
# #     noiselist.append(TimeSeries.read(f'/media/daniel/easystore/Daniel/MSc-files/Noise/ts-reference-{ifo}',
# #                                      format='hdf5').resample(sampling_frequency))
# #     asds.append(noise_ifo.asd(fftlength=4).interpolate(df=1).crop(start=21))  # I don't like this
#
#
# seed = 0
# # for seed in range(60, 120):
# bilby.core.utils.random.seed(seed)
#
# waveform_arguments = dict(
#     waveform_approximant="IMRPhenomXPHM",
#     minimum_frequency=min_freq,
#
# )
#
# waveform_generator = bilby.gw.WaveformGenerator(
#     duration=duration,
#     sampling_frequency=sampling_frequency,
#     frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#     parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#     waveform_arguments=waveform_arguments,
#
# )
#
# injection_kwargs = {
#     # Always needed (timeseries vs q-transform)
#     'sampling_frequency': sampling_frequency,
#     'waveform_generator': waveform_generator,
#
#     # Image-only arguments
#     'img_res': (128, 128),  # (64, 96),  # (height, width) in pixels
#     'duration': duration,
#
#     'qtrans_kwargs': {
#         'frange': (min_freq, 300),
#         'qrange': (4, 64),  # Not changing much, apparently
#         'outseg': (-0.1, 0.1)
#     }
# }
# resol = injection_kwargs['img_res']
#
# prior = bilby.gw.prior.BBHPriorDict()
# # prior['luminosity_distance'] = 100.  # in Mpc. Set to 100 Mpc so it can be scaled according to target SNR
# # print(*prior.items(), sep='\n')
# # DONE: attempt different strategy. If SNR not within marks, resample whole prior and try again
# data = get_data(N=12,
#                 targ_snrs=[5, 40],
#                 datatype='q-trans',
#                 ifos=ifolist,
#                 # asds=asds,
#                 # parallel=True,
#                 use_tqdm=True,
#                 return_type='dict',
#                 seed=seed,
#                 joblib_kwargs=joblib_kwargs,
#                 **injection_kwargs
#                 )
# # pprint(data[0])
# for i in range(12):
#     plt.imshow(make_image(data[i]), aspect=resol[1]/resol[0])
#     pprint(data[i]['parameters'])
#     print('\n'*2)
#     plt.show()
# # torch.save(data, files_path / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
# # print(f'Saved dataset {seed}')

'''
000 to 059: [5, 40] SNR on default noise
060 to 063: [5, 40] SNR on 2017 noise (no clear signals, seems to be trash)
'''

'''
I think the problem with this pipeline is the lack of semi-random noise
'''

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
noise_dir = files_dir / 'Noise' / 'MKIII'

'''
1185366342: 2017-07-29 12:25:24     V1 not available, full LIGO
1187058342: 2017-08-17 02:25:24
1186295142: 2017-08-09 06:25:24     H1 not available earlier
1187490342: 2017-08-23 02:25:24     
1240194342: 2019-04-25 02:25:24
    1249784742: 2019-08-14 02:25:24
1263090342: 2020-01-15 02:25:24     H1 has a glitch, V1 gives no psd
1267928742: 2020-03-11 02:25:24
'''
import matplotlib.pyplot as plt

# times = [
#     1187058342,
#     1249784742,
#     # 1240194342,
#     # 1263090342,
#     1267928742,
# ]
times = [1185366342, ]
colors = ['tab:blue', 'tab:orange', 'tab:green']
linestyles = ['-', '--', ':']

for t in times:
    for ifo in ('L1', 'H1', 'V1'):
        strain = TimeSeries.fetch_open_data(ifo, t, t+500, verbose=True)
        strain.write(target=noise_dir/f'noise_{t}_{ifo}', format='hdf5')


# for t, lst in zip(times, linestyles):
#     for ifo, color in zip(('L1', 'H1', 'V1'), colors):
#         # strain = TimeSeries.fetch_open_data(ifo, t, t+500, verbose=True)
#         # strain.write(target=noise_dir/f'noise_{t}_{ifo}', format='hdf5')
#         asd = TimeSeries.read(noise_dir/f'noise_{t}_{ifo}').asd(fftlength=4)
#         # plot = strain.plot()
#         # plot.show()
#         plt.loglog(asd.frequencies, asd.value, label=f'{t}-{ifo}', color=color, linestyle=lst)
#
# plt.xlim((10, 1000))
# plt.legend()
# plt.show()
