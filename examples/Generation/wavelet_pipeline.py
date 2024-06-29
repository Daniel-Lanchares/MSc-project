# import numpy as np
import bilby
import torch
import matplotlib.pyplot as plt
from pprint import pprint

from dtempest.gw.conversion import make_image
from dtempest.gw.generation.timeseries import get_data
# from dtempest.gw.generation.images import dataset_q_transform
# from dtempest.gw.generation.wavelet import Morlet, CWT

import warnings

bilby.utils.logging.disable(level=50)
warnings.filterwarnings("ignore")  # Ignore the annoying default warnings

# %%
duration = 2.0  # seconds
sampling_frequency = 1024.0  # Hz
min_freq = 20.0  # Hz

seed = 42
bilby.core.utils.random.seed(seed)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    minimum_frequency=min_freq,

)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,

)

injection_kwargs = {
    'datatype': 'q-trans',
    # Always needed (timeseries vs q-transform)
    'sampling_frequency': sampling_frequency,
    'waveform_generator': waveform_generator,

    # Image-only arguments
    'img_res': (64, 96),
    'duration': duration,

    'qtrans_kwargs': {
        'frange': (min_freq, 300),
        'qrange': (4, 64),  # Not doing much, apparently
        'outseg': (-0.1, 0.1)
    }
}
resol = injection_kwargs['img_res']

prior = bilby.gw.prior.BBHPriorDict()
prior['luminosity_distance'] = 100.  # in Mpc. Set to 100 Mpc so it can be scaled according to target SNR
# print(*prior.items(), sep='\n')

# data, params = get_data(targ_snrs=[5, 40],
#                         N=12,  # 128
#                         parallel=True,
#                         use_tqdm=True,
#                         waveform_generator=waveform_generator,
#                         )
# batch = np.stack(data)
# pprint(params[:2])
# pprint(data[:2])
#
# getcwt = CWT(dt=1 / 1024, fmin=20, fmax=512, trainable=False, hop_length=20, wavelet=Morlet())
# spec_batch = getcwt(torch.from_numpy(batch))
#
# print(batch.shape)  # n_images, n_channels, length
# print(spec_batch.shape)  # n_images, n_channels, height, width
#
# from skimage.transform import resize
#
# # %%
# # plt.pcolormesh(spec_batch[23, 0][:, :].flip(0))
# plt.imshow(resize(make_image(spec_batch[0]), (128, 128)))
# plt.show()

data = get_data(targ_snrs=[5, 40],
                N=12,  # 128
                parallel=True,
                use_tqdm=True,
                return_type='dict',
                seed=seed,
                **injection_kwargs
                )
# data = dataset_q_transform(data, resol, **qtrans_kwargs)
# pprint(data[0])
# print(data[0])
# print(make_image(data[0]).shape)
# for i in range(12):
#     plt.imshow(make_image(data[i]), aspect=resol[1]/resol[0])
#     pprint(data[i]['parameters'])
#     print('\n'*2)
#     plt.show()
torch.save(data, files_path / f'Raw_Dataset_{seed:0{zero_pad}}.pt')