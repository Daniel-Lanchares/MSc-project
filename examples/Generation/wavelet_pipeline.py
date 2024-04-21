import numpy as np
import bilby
import torch
import matplotlib.pyplot as plt
from pprint import pprint

from dtempest.gw.conversion import make_image
from dtempest.gw.generation.wavelet_gen import Morlet, CWT, get_data

import warnings

bilby.utils.logging.disable(level=50)
warnings.filterwarnings("ignore")  # Ignore the annoying default warnings

# %%
duration = 2.0  # seconds
sampling_frequency = 1024.0  # Hz
np.random.seed(42)

waveform_arguments = dict(
    waveform_approximant="IMRPhenomXPHM",
    minimum_frequency=20.0,

)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,

)

prior = bilby.gw.prior.BBHPriorDict()
prior['luminosity_distance'] = 100.  # in Mpc. Set to 100 Mpc so it can be scaled according to target SNR
# print(*prior.items(), sep='\n')

data, params = get_data(targ_snrs=[5, 40],
                        N=12,  # 128
                        parallel=True,
                        use_tqdm=True,
                        waveform_generator=waveform_generator,
                        )
batch = np.stack(data)
pprint(params)

getcwt = CWT(dt=1 / 1024, fmin=20, fmax=512, trainable=False, hop_length=20, wavelet=Morlet())
spec_batch = getcwt(torch.from_numpy(batch))

print(batch.shape)  # n_images, n_channels, length
print(spec_batch.shape)  # n_images, n_channels, height, width

from skimage.transform import resize

# %%
# plt.pcolormesh(spec_batch[23, 0][:, :].flip(0))
plt.imshow(resize(make_image(spec_batch[0]), (128, 128)))
plt.show()
