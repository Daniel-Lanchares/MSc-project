"""
Basic generation script

On my end its is being quite slow because multiprocessing only kicks in if I execute directly on the mkiii.py script
"""

import torch
import numpy as np
from pathlib import Path
from functools import partial

import bilby
from gwpy.timeseries import TimeSeries

from dtempest.gw.generation import generate


def quick_show(data, n=12):
    from pprint import pprint
    import matplotlib.pyplot as plt

    from dtempest.gw.conversion import make_image
    for i in range(n):
        plt.imshow(make_image(data[i]), aspect=resol[1] / resol[0])
        pprint(data[i]['parameters'])  # ['_snr']
        # print(data[i]['parameters']['luminosity_distance'])
        # print(data[i]['parameters']['theta_jn'])
        print('\n' * 2)
        plt.show()

if __name__ == '__main__':
    # Arguments for parallelization utilities.
    # See joblib's Parallel for more info.
    parallel_kwargs = {
            'n_jobs': 11,
            'backend': 'loky'#'multiprocessing'
        }

    duration = 4.0  # seconds  # change from around 2 to 8 depending on masses
    sampling_frequency = 1024  # Hz  DON'T KNOW WHY (yet), BUT DO NOT CHANGE!
    min_freq = 20.0  # Hz

    # List of interferometers (R, G, B)
    ifolist = ('L1', 'H1', 'V1')
    snr_range = (5, np.inf)


    # Padding depending on the amount of chunks in which to split the dataset
    zero_pad = 3
    def base_id_format(j, zero_pad, seed):
        return f'{seed:0{zero_pad}}.{j:05}'

    imgs_per_seed = 1000
    n_seeds = 5 # change for bigger datasets (currently 5k)
    train_seeds = list(range(n_seeds))
    val_seeds = list(range(900, 901)) # (1k validation)

    files_dir = Path('')           # Main directory
    rawset_dir = files_dir / 'Raw Datasets' # Data directory
    noise_dir = files_dir / 'Noise'         # Noise directory

    # Each chunk of the dataset is generated with noise (ASD) from one of these times
    times = [# Need to gather the noise first, then select available times

        # 1126257941,  # Close to GW150914, V1 missing
        # 1185366342,  # Close to GW170729, V1 missing
        # 1186295142,  # Close to GW170809 PROBLEMATIC, H1 is zeros?
        # 1187058342,  # Close to GW170818
        # 1187490342,  # Close to GW170823
        # 1187528242,  # Even closer
        # 1240194342,
        # 1249784742,
        # 1263090342, # Missing one
        1267928742,
    ]

    # Noise dict construction
    asd_dict = {
        t: [TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1).value for ifo in ifolist]
        for t in times}

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
        'img_res': (50, 200),  # (128, 128) for GP14, (48, 72) for LH14,  # (height, width) in pixels
        'duration': duration,

        'qtrans_kwargs': {
            'frange': (min_freq, 300),
            'qrange': (4, 64),  # Not changing much, apparently
            'outseg': (-0.3, 0.1) # Important to tweak with quick_show and small datasets to ensure signal is in frame
        }
    }
    resol = injection_kwargs['img_res']
    prior = bilby.gw.prior.BBHPriorDict()

    # Various prior changes from the bilby default

    # prior['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(
    #     name='chirp_mass', minimum=20, maximum=60)  # 30 to 120 for high mass (25 to 100 normal)
    # prior['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(
    #     name='mass_ratio', minimum=0.5, maximum=1,)  # equal_mass=True) favours higher qs
    # prior['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=20, maximum=80)
    # prior['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=20, maximum=80)
    # prior['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.88)
    # prior['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.88)
    prior['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(
        name='luminosity_distance', minimum=1e2, maximum=5e3)
    # del prior['chirp_mass'], prior['mass_ratio']
    # tc = 1187529256.5  # GW170823
    # prior['geocent_time'] = bilby.gw.prior.Uniform(name="geocent_time", minimum=tc-0.05, maximum=tc+0.05)
    if 'geocent_time' not in prior.keys():
        prior['geocent_time'] = 0.0

    # Personal experiment with mod 24h normalized time
    # prior['normalized_time'] = bilby.gw.prior.Uniform(name='normalized_time', minimum=0.0, maximum=1.0, boundary='periodic')


    for seed in train_seeds + val_seeds:
        id_format = partial(base_id_format, zero_pad=zero_pad, seed=seed)

        bilby.core.utils.random.seed(seed)
        t = bilby.core.utils.random.rng.choice(times)
        print(f'Chosen time: {t}')
        asds = asd_dict[t]




        data = generate(n=imgs_per_seed,
                        snr_range=snr_range,
                        ifos=ifolist,
                        prior=prior,
                        seed=seed,
                        datatype='q-trans',
                        asds=asds,
                        id_format=id_format,
                        parallel=True,
                        joblib_kwargs=parallel_kwargs,
                        **injection_kwargs)



        data.metadata['noise_tGPS'] = t
        torch.save(data, rawset_dir / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
        print(f'Saved dataset {seed}')

        # This can be ignored in a cluster
        # if seed % 5 == 0:
        #     print('Resting for a bit to avoid exploding')
        #     time.sleep(30)