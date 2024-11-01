# from pprint import pprint

from typing import Callable

import bilby
bilby.utils.logging.disable(level=50)
import numpy as np
from bilby.core.prior import PriorDict
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
# from matplotlib import pyplot as plt
from tqdm import tqdm

from dtempest.gw.conversion import make_image
# from dtempest.gw.conversion import make_image
from dtempest.gw.generation.generation_utils import construct_dict

default_jlib_kw = {
    'n_jobs': 12,
    'backend': 'multiprocessing'}


def generate(n: int = 1,
             snr_range: tuple[float, float] | int | str = (5, 40),
             prior: PriorDict = None,
             ifos: tuple[str, ...] = None,
             asds: list[list] | list = None,
             seed: int = 0,
             datatype: str = 'timeseries',
             id_format: Callable = None,
             parallel: bool = False,
             joblib_kwargs: dict = None,
             **inj_func_kwargs
             ):
    bilby.utils.logging.disable()
    bilby.core.utils.random.seed(seed)
    snr_range = -1 if snr_range == 'zero_noise' else snr_range
    # asds = handle_asds(asds, n)
    datatype, inj_func = handle_datatype(datatype)

    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()
    # if 'geocent_time' not in prior.keys():
    #     prior['geocent_time'] = 0.
    prior_samples = [prior.sample() for _ in range(n)]

    if ifos is None:
        ifos = ["H1", "L1", "V1"]

    if id_format is None:
        def id_format(j): return f'{seed}.{j:05}'

    if parallel:
        if joblib_kwargs is None:
            joblib_kwargs = {}
        from copy import copy
        _joblib_kwargs = copy(default_jlib_kw)
        _joblib_kwargs.update(joblib_kwargs)
        out = Parallel(**_joblib_kwargs)(
            delayed(inj_func)(snr_range,
                              prior_s,
                              prior,
                              ifolist=ifos,
                              asds=asds,
                              **inj_func_kwargs)
            for prior_s in tqdm(prior_samples, desc=f'Creating Raw Dataset {seed}'))
    else:
        out = [inj_func(snr_range,
                        prior_s,
                        prior,
                        ifolist=ifos,
                        asds=asds,
                        **inj_func_kwargs)
               for prior_s in tqdm(prior_samples, desc=f'Creating Raw Dataset {seed}')]

    metadata = {
        'prior': prior,
        'snr_range': snr_range,
        'ifos': ifos,
        'datatype': datatype,
        'seed': seed,
        'inject_per_seed': n,
        'injection_kwargs': {key: value for key, value in inj_func_kwargs.items() if key != 'waveform_generator'}
    }

    return construct_dict(data=out, ifos=ifos, datatype=datatype, id_format=id_format, metadata=metadata)


def handle_time(prior_samples: dict):
    if 'geocent_time' in prior_samples.keys():
        time = prior_samples['geocent_time']
    if 'normalized_time' in prior_samples.keys():
        time = prior_samples['normalized_time']*24*3600
        prior_samples['geocent_time'] = time
    else:
        time = 0.0
    return time


# def handle_asds(asds, m: int):
#     # If its None return [None, ...] and if it is [a1, a2,...] return [[a1, a2,...], ...]
#     if asds is None or not all(isinstance(elem, list) for elem in asds):
#         return [asds for _ in range(m)]
#     else:
#         return asds


def handle_datatype(datatype) -> tuple[str, Callable]:
    if datatype == 'timeseries':
        return datatype, generate_timeseries
    elif datatype in ['q-transform', 'q_transform', 'q-trans', 'q_trans', 'qtrans']:
        datatype = 'q-transforms'
        from dtempest.gw.generation.images import generate_q_transforms
        return datatype, generate_q_transforms
    elif datatype == 'wavelet':
        raise NotImplementedError
    else:
        raise ValueError(f"datatype '{datatype}' not understood. "
                         f"Possibilities are 'timeseries', 'q-transform' and 'wavelet'")


def generate_timeseries(snr_range: tuple[float, float] | int,
                        prior_samples,
                        prior: PriorDict,
                        ifolist,
                        asds,
                        sampling_frequency=1024,
                        waveform_generator=None,
                        ):
    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    duration = waveform_generator.duration

    # Initialize a counter variable.
    iterator = 0

    time = handle_time(prior_samples)

    # Set up the two interferometers.
    ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds,
                      time=time)

    # Save the background frequency domain strain data of the two interferometers if you always want the same noise.
    # ifos_bck = [ifo.frequency_domain_strain for ifo in ifos]

    if snr_range == -1:
        # Inject the signal into the noise-free background and return the resulting strain data.
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                duration=duration,
                                                start_time=time - duration / 2)
        # prior_samples['geocent_time'] = 0.
        ifos.inject_signal(prior_samples, waveform_generator=waveform_generator)
        return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_samples
    else:
        if snr_range == 0:  # Return as is
            # ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            # ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            ifos.inject_signal(prior_samples, waveform_generator=waveform_generator)
            return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_samples
        # Iteratively inject the signal and adjust the luminosity distance until the
        # injected signal has a SNR close to the target SNR.
        while True:
            ifos.inject_signal(prior_samples, waveform_generator=waveform_generator)
            new_snr = getsnr(ifos)
            if snr_range[0] < new_snr < snr_range[1]:
                # Return the injected strain data for each interferometer.
                # print(f'Worked in {iterator+1} iterations')
                prior_samples['_snr'] = new_snr
                return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_samples
            bilby.core.utils.random.seed(int(new_snr * 1e+6))

            prior_samples = prior.sample()
            time = handle_time(prior_samples)

            ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds,
                      time=time)
            iterator += 1
            # Inject the signal into the background strain data and update the geocentric time of the signal.
            # ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            # for i, ifo in enumerate(ifos_):
            #     # This would always give the same noise
            #     ifo.set_strain_data_from_frequency_domain_strain(ifos_bck[i], sampling_frequency=sampling_frequency,
            #                                                      duration=duration, start_time=duration / 2)
            # prior_samples['geocent_time'] = 0.
            # ifos_.inject_signal(prior_samples, waveform_generator=waveform_generator)

            # Check if the SNR of the injected signal is close to the target SNR.
            # new_snr = getsnr(ifos_)
            # print(new_snr)
            # if np.isnan(new_snr):
            #     new_snr = bilby.core.utils.random.rng.uniform()  # Since < 1 it should never pass a test
            # print(new_snr)
            # print(new_snr)
            # pprint(prior_samples)
            # if snr_range[0] < new_snr < snr_range[1]:
            #     # Return the injected strain data for each interferometer.
            #     # print(f'Worked in {iterator+1} iterations')
            #     prior_samples['_snr'] = new_snr
            #     return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_samples

            # Adjust the luminosity distance of the signal by the ratio of the current SNR to the target SNR.
            # snr_ratios = np.array(new_snr / targ_snr)
            # prior_s['luminosity_distance'] = prior_s['luminosity_distance'] * np.prod(snr_ratios)

            # To avoid repetition of samples when calculating in parallel
            # Multiplied by a million to avoid fixed points (gets pipeline stuck)
            # bilby.core.utils.random.seed(int(new_snr*1e+6))
            # prior_samples = prior.sample()
            # iterator += 1
            # print(f'iteration {iterator+1}, {new_snr}')


def setup_ifos(ifos=None, asds=None, duration=None, sampling_frequency=None, time=0.):
    """Sets up the two interferometers, H1 and L1. Uses ALIGO power spectral density by default.

    Args:
        ifos: A list of interferometer names
        asds: An optional list of amplitude spectral densities for each interferometer
        duration: The duration of the injected signal.
        sampling_frequency: The sampling frequency of the injected signal.

    Returns:
        A list of two bilby.gw.detector.Interferometer objects.
    """
    if ifos is None:
        ifos = ["H1", "L1"]

    # Create a list of two bilby.gw.detector.Interferometer objects.
    ifos = bilby.gw.detector.InterferometerList(ifos)

    # Set the amplitude spectral density of each interferometer if given
    if asds is not None:
        from bilby.gw.detector.psd import PowerSpectralDensity
        for i, ifo in enumerate(ifos):
            ifo_frequency_array = np.fft.rfftfreq(sampling_frequency, 1 / sampling_frequency)
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array > ifo.minimum_frequency]
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array < ifo.maximum_frequency]

            # ifo.power_spectral_density = \
            #     PowerSpectralDensity.from_amplitude_spectral_density_array(frequency_array=ifo_frequency_array,
            #                                                                asd_array=asds[i])

            # Depends on whether sampling_rate/2 is larger than the interferometer's max frequency
            max_freq = min([ifo.maximum_frequency, ifo_frequency_array[-1]])
            asd_array = asds[i][int(ifo.minimum_frequency):int(max_freq)]
            ifo.power_spectral_density = PowerSpectralDensity(frequency_array=ifo_frequency_array,
                                                              asd_array=asd_array)
    # Set the strain data for each interferometer from a power spectral density.
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=time - duration / 2,
    )

    return ifos


def getsnr(ifos):
    """Calculates the signal-to-noise ratio (SNR) of the injected signal in two interferometers.

    Args:
        ifos: A list of two bilby.gw.detector.Interferometer objects.

    Returns:
        A float, representing the SNR of the injected signal.
    """

    # Calculate the matched filter SNR of the injected signal in each interferometer.
    # matched_filter_snr_h1 = abs(ifos[0].meta_data['matched_filter_SNR'])
    # matched_filter_snr_l1 = abs(ifos[1].meta_data['matched_filter_SNR'])
    # print(*ifos, sep="\n")

    matched_filters_sq = [abs(ifo.meta_data['matched_filter_SNR']) ** 2 for ifo in ifos]

    # Calculate the total SNR of the injected signal by combining the matched filter SNRs of the two interferometers.
    snr = np.sqrt(np.sum(matched_filters_sq))

    return np.nan_to_num(snr)


if __name__ == '__main__':
    # bilby.utils.logging.disable(level=50)
    # import warnings
    # warnings.filterwarnings("ignore")  # Ignore the annoying default warnings

    parallel_kwargs = {
        'n_jobs': 11,
        'backend': 'loky'#'multiprocessing'
    }

    duration = 4.0  # seconds  # 4.0 for low mass, normal 2 seconds
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

    from pathlib import Path
    from gwpy.timeseries import TimeSeries

    zero_pad = 3


    def id_format(j):
        return f'{seed:0{zero_pad}}.{j:05}'

    snr_range = (7.5, np.inf)
    files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
    rawset_dir = files_dir / 'Raw Datasets' / 'long-window+normal-time-test'
    noise_dir = files_dir / 'Noise' / 'MKIII'
    times = [
        # 1126257941,  # Close to GW150914, V1 missing
        # 1185366342,  # Close to GW170729, V1 missing
        # 1186295142,  # Close to GW170809 PROBLEMATIC, H1 is zeros?
        1187058342,  # Close to GW170818
        # 1187490342,  # Close to GW170823
        # 1187528242,  # Even closer
        1240194342,
        1249784742,
        # 1263090342, # Missing one
        1267928742,
    ]
    asd_dict = {
        t: [TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1).value for ifo in ifolist]
        for t in times}

    # for asds in asd_dict.values():
    #     for asd in asds:
    #         plt.loglog(range(len(asd)), asd)
    # plt.show()

    # t = times[0]
    # for ifo in ifolist:
    #     asd = TimeSeries.read(noise_dir / f'noise_{t}_{ifo}').asd(fftlength=4).interpolate(df=1)
    #     asds.append(asd.value)

    # seed = 0
    for seed in list(range(100))+[999,]:#range(990, 1000):
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
            'img_res': (50, 200),  # (32, 48),  # (height, width) in pixels
            'duration': duration,

            'qtrans_kwargs': {
                'frange': (min_freq, 300),
                'qrange': (4, 64),  # Not changing much, apparently
                'outseg': (-0.3, 0.1)
            }
        }
        resol = injection_kwargs['img_res']
        prior = bilby.gw.prior.BBHPriorDict()

        # prior['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(
        #     name='chirp_mass', minimum=20, maximum=60)  # 30 to 120 for high mass (25 to 100 normal)
        # prior['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(
        #     name='mass_ratio', minimum=0.5, maximum=1,)  # equal_mass=True) favours higher qs
        # prior['mass_1'] = bilby.gw.prior.Uniform(name='mass_1', minimum=20, maximum=80)
        # prior['mass_2'] = bilby.gw.prior.Uniform(name='mass_2', minimum=20, maximum=80)
        # prior['a_1'] = bilby.gw.prior.Uniform(name='a_1', minimum=0, maximum=0.88)
        # prior['a_2'] = bilby.gw.prior.Uniform(name='a_2', minimum=0, maximum=0.88)
        prior['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(
            name='luminosity_distance', minimum=1e2, maximum=5e3)  # to 6e3 for high mass (1e2 to 6e3 normal)
        # del prior['chirp_mass'], prior['mass_ratio']
        # tc = 1187529256.5  # GW170823
        # prior['geocent_time'] = bilby.gw.prior.Uniform(name="geocent_time", minimum=tc-0.05, maximum=tc+0.05)
        # prior['geocent_time'] = 0.0
        # prior['normalized_time'] = bilby.gw.prior.Uniform(name='normalized_time', minimum=0.0, maximum=1.0, boundary='periodic')

        n = 1000

        data = generate(n=n,
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
        # from pprint import pprint
        # for i in range(n):
        #     plt.imshow(make_image(data[i]), aspect=resol[1] / resol[0])
        #     pprint(data[i]['parameters'])  # ['_snr']
        #     # print(data[i]['parameters']['luminosity_distance'])
        #     # print(data[i]['parameters']['theta_jn'])
        #     print('\n' * 2)
        #     plt.show()

        import torch
        import time
        data.metadata['noise_tGPS'] = t
        torch.save(data, rawset_dir / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
        print(f'Saved dataset {seed}')

        if seed % 5 == 0:
            print('Resting for a bit to avoid exploding')
            time.sleep(30)
