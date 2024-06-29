# import os
import time
import torch
import numpy as np
from pathlib import Path
# from matplotlib import pyplot as plt

import bilby
from gwpy.timeseries import TimeSeries

# from dtempest.gw.conversion import make_image
from dtempest.gw.generation import generate


#
# zero_pad = 3
#
# joblib_kwargs = {
#     'n_jobs': 12,
#     'backend': 'multiprocessing'
# }
#
# duration = 2.0  # seconds
# sampling_frequency = 1024  # Hz  DON'T KNOW WHY, BUT DO NOT MESS WITH IT! (I think I found why, but better not touch)
# min_freq = 20.0  # Hz
#
# ifolist = ('L1', 'H1', 'V1')
# # asds = []
#
# files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
# rawset_dir = files_dir / 'Raw Datasets' / 'MKIII pipe'
# noise_dir = files_dir / 'Noise' / 'MKIII'
# times = [
#     1187058342,
#     1249784742,
#         # 1240194342,
#         # 1263090342,
#     1267928742,
# ]
# asd_dict = {
#     t: [TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1).value for ifo in ifolist]
#     for t in times}
#
# # seed = 0
# for seed in range(60):
#     bilby.core.utils.random.seed(seed)
#
#     t = bilby.core.utils.random.rng.choice(times)
#     print(f'Chosen time: {t}')
#     # for ifo in ifolist:
#     #     asd = TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1)
#     #     asds.append(asd.value)
#     asds = asd_dict[t]
#
#     waveform_arguments = dict(
#         waveform_approximant="IMRPhenomXPHM",
#         minimum_frequency=min_freq,
#
#     )
#
#     waveform_generator = bilby.gw.WaveformGenerator(
#         duration=duration,
#         sampling_frequency=sampling_frequency,
#         frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
#         parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
#         waveform_arguments=waveform_arguments,
#
#     )
#     injection_kwargs = {
#         # Always needed (timeseries vs q-transform)
#         'sampling_frequency': sampling_frequency,
#         'waveform_generator': waveform_generator,
#
#         # Image-only arguments
#         'img_res': (128, 128),  # (64, 96),  # (height, width) in pixels
#         'duration': duration,
#
#         'qtrans_kwargs': {
#             'frange': (min_freq, 300),
#             'qrange': (4, 64),  # Not changing much, apparently
#             'outseg': (-0.1, 0.1)
#         }
#     }
#     resol = injection_kwargs['img_res']
#     prior = bilby.gw.prior.BBHPriorDict()
#
#     n = 1000
#
#     data = generate(n=n,
#                     snr_range=(5, np.inf),
#                     ifos=ifolist,
#                     seed=seed,
#                     datatype='q-trans',
#                     asds=asds,
#                     parallel=True,
#                     joblib_kwargs=joblib_kwargs,
#                     **injection_kwargs)
#     # for i in range(n):
#     #     plt.imshow(make_image(data[i]), aspect=resol[1] / resol[0])
#     #     print(data[i]['parameters']['_snr'])
#     #     print(data[i]['parameters']['luminosity_distance'])
#     #     print(data[i]['parameters']['theta_jn'])
#     #     print('\n' * 2)
#     #     plt.show()
#     torch.save(data, rawset_dir / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
#     print(f'Saved dataset {seed}')


def main():
    joblib_kwargs = {
        'n_jobs': 12,
        'backend': 'multiprocessing'
    }

    duration = 2.0  # seconds
    sampling_frequency = 1024  # Hz  DON'T KNOW WHY, BUT DO NOT MESS WITH IT!
    min_freq = 20.0  # Hz

    ifolist = ('L1', 'H1', 'V1')
    # asds = []
    # noiselist = []
    # for ifo in ifolist:
    #     Read noise from file, notch_filter and resample to 2048 Hz
    #     noiselist.append(TimeSeries.read(f'/media/daniel/easystore/Daniel/MSc-files/Noise/ts-reference-{ifo}',
    #                                      format='hdf5').resample(sampling_frequency))
    #     asds.append(noise_ifo.asd(fftlength=4).interpolate(df=1).crop(start=21))  # I don't like this

    files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
    rawset_dir = files_dir / 'Raw Datasets' / 'mid range' / 'extra low res'
    noise_dir = files_dir / 'Noise' / 'MKIII'
    times = [
        1187058342,
        1249784742,
        # 1240194342,
        # 1263090342,
        1267928742,
    ]
    asd_dict = {
        t: [TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1).value for ifo in ifolist]
        for t in times}

    # t = times[0]
    # for ifo in ifolist:
    #     asd = TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1)
    #     asds.append(asd.value)

    # seed = 0
    for seed in range(120):
        bilby.core.utils.random.seed(seed)
        t = bilby.core.utils.random.rng.choice(times)
        print(f'Chosen time: {t}')
        asds = asd_dict[t]

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
            # Always needed (timeseries vs q-transform)
            'sampling_frequency': sampling_frequency,
            'waveform_generator': waveform_generator,

            # Image-only arguments
            'img_res': (48, 72),  # (height, width) in pixels
            'duration': duration,

            'qtrans_kwargs': {
                'frange': (min_freq, 300),
                'qrange': (4, 64),  # Not changing much, apparently
                'outseg': (-0.1, 0.1)
            }
        }
        # resol = injection_kwargs['img_res']
        prior = bilby.gw.prior.BBHPriorDict()

        prior['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(
            name='chirp_mass', minimum=20, maximum=60)  # 30 to 120 for high mass (25 to 100 normal)

        n = 1000

        data = generate(n=n,
                        snr_range=(5, np.inf),
                        ifos=ifolist,
                        seed=seed,
                        datatype='q-trans',
                        asds=asds,
                        parallel=True,
                        joblib_kwargs=joblib_kwargs,
                        **injection_kwargs)
        # for i in range(n):
        #     plt.imshow(make_image(data[i]), aspect=resol[1] / resol[0])
        #     print(data[i]['parameters']['_snr'])
        #     print(data[i]['parameters']['luminosity_distance'])
        #     print(data[i]['parameters']['theta_jn'])
        #     print('\n' * 2)
        #     plt.show()
        zero_pad = 4
        data.metadata['noise_tGPS'] = t
        torch.save(data, rawset_dir / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
        print(f'Saved dataset {seed}')

        if seed % 5 == 0:
            print('Resting for a bit to avoid exploding')
            time.sleep(30)


if __name__ == '__main__':
    main()
