# Core modules
import numpy as np
from tqdm import tqdm
from typing import Callable
from joblib import Parallel, delayed

# GW-modules
import bilby
from bilby.gw.detector.psd import PowerSpectralDensity

# Internal dependencies
from dtempest.core.common_utils import identity

from dtempest.gw.generation.generation_utils import construct_dict

"""
Derived from a script provided by Osvaldo Freitas

Generation of timeseries data to be later converted to images
"""


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

    return snr


def setup_ifos(ifos=None, asds=None, duration=None, sampling_frequency=None):
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
        for i, ifo in enumerate(ifos):
            ifo_frequency_array = np.fft.rfftfreq(sampling_frequency, 1 / sampling_frequency)
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array > ifo.minimum_frequency]
            ifo_frequency_array = ifo_frequency_array[ifo_frequency_array < ifo.maximum_frequency]

            ifo.power_spectral_density = \
                PowerSpectralDensity.from_amplitude_spectral_density_array(ifo_frequency_array,
                                                                           asds[i])
    # Set the strain data for each interferometer from a power spectral density.
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=0 - duration / 2,
    )

    return ifos


def get_inj_ifos_ts(targ_snr,
                    prior_s,
                    sampling_frequency=1024,
                    waveform_generator=None,
                    ifolist=None,
                    noise_ts=None,
                    asds=None) -> tuple[list[np.ndarray], dict]:
    """Injects a signal into two interferometers and returns the injected strain data.

    Args:
        targ_snr: The target signal-to-noise ratio (SNR). If targ_snr is equal to -1,
            the function will inject the signal into a noise-free background.
        prior_s: Prior sample to use in generation.
        sampling_frequency: The sampling frequency of the injected signal.
        waveform_generator: The waveform generator to use for the injected signal.
        ifolist: List of interferometers. Defaults to LIGO-VIRGO collaboration.

    Returns:
        A tuple of two NumPy arrays, containing the injected strain data of the two
        interferometers.
    """

    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    duration = waveform_generator.duration

    # if prior is None:
    #     prior = bilby.gw.prior.BBHPriorDict()
    if ifolist is None:
        ifolist = ["L1", "H1", "V1"]

    # Initialize a counter variable.
    iterator = 0

    # Set up the two interferometers.
    if noise_ts is None:
        ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
    else:
        # assert isinstance(noise_ts, gwpy.timeseries.TimeSeries), (f"noise strain is not a gwpy.Timeseries object, "
        #                                                           f"type: '{type(noise_ts)}'")
        # ifos = bilby.gw.detector.InterferometerList(ifolist)
        # for ifo, noise in zip(ifos, noise_ts):
        #     # tau = bilby.core.utils.random.rng.uniform(duration/2, noise.duration.value-duration/2)
        #     # print(tau)
        #     # noise_slice = noise.crop(tau-duration/2, duration/2, copy=True)
        #     ifo.set_strain_data_from_gwpy_timeseries(noise)
        raise NotImplementedError

    # Save the background frequency domain strain data of the two interferometers.
    ifos_bck = [ifo.frequency_domain_strain for ifo in ifos]

    # Sample a set of prior parameters for the signal, including the luminosity distance.
    # prior_s = prior.sample()

    # Set the initial guess for the luminosity distance.
    # prior_s['luminosity_distance'] = 1000
    # Kept mostly for debugging purposes
    prior_s['_snr'] = targ_snr

    if targ_snr == -1:
        # Inject the signal into the noise-free background and return the resulting strain data.
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                duration=duration, start_time=duration / 2)
        prior_s['geocent_time'] = 0.
        ifos.inject_signal(prior_s, waveform_generator=waveform_generator)
        return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos], prior_s
    else:
        if targ_snr == 0:
            # ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_s
        # Iteratively inject the signal and adjust the luminosity distance until the
        # injected signal has a SNR close to the target SNR.
        while True:
            # Inject the signal into the background strain data and update the geocentric time of the signal.
            ifos_ = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
            for i, ifo in enumerate(ifos_):
                ifo.set_strain_data_from_frequency_domain_strain(ifos_bck[i], sampling_frequency=sampling_frequency,
                                                                 duration=duration, start_time=duration / 2)
            prior_s['geocent_time'] = 0.
            ifos_.inject_signal(prior_s, waveform_generator=waveform_generator)

            # Check if the SNR of the injected signal is close to the target SNR.
            new_snr = getsnr(ifos_)
            if abs(new_snr - targ_snr) < 1.5:
                # Return the injected strain data for each interferometer.
                # print(f'Worked in {iterator+1} iterations')
                prior_s['_snr'] = new_snr
                return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_s

            # Adjust the luminosity distance of the signal by the ratio of the current SNR to the target SNR.
            snr_ratios = np.array(new_snr / targ_snr)
            prior_s['luminosity_distance'] = prior_s['luminosity_distance'] * np.prod(snr_ratios)
            iterator += 1


def handle_noise(noise_list, strain_duration, N):
    rng = bilby.core.utils.random.rng
    taus = rng.uniform(strain_duration/2, min([noise.duration.value for noise in noise_list])-strain_duration/2, N)
    result = [
        [ifo_noise.crop(ifo_noise.t0.value + tau-strain_duration/2, ifo_noise.t0.value + tau+strain_duration/2,
                        copy=True) for ifo_noise in noise_list]
        for tau in taus]
    # print(result)
    return result


def handle_snr(targ_snrs, N):
    if targ_snrs is None:
        targ_snrs = []
    rng = bilby.core.utils.random.rng
    if isinstance(targ_snrs, str) or isinstance(targ_snrs, int):
        assert N is not None, 'Number of injections must be given if using single SNR value'
        if targ_snrs == 'zero_noise' or targ_snrs == -1:

            targ_snrs = -1 * np.ones(N)  # an SNR of -1 passed to get_inj_ifos will result in a "clean" injection
        else:
            targ_snrs = np.ones(N) * targ_snrs

    else:
        if len(targ_snrs) == 2:  # Check if range was given
            assert N is not None, 'Number of injections must be given if using SNR range'
            print(f'SNR interval given. Sampling {N} samples uniformly from range [{min(targ_snrs)}, {max(targ_snrs)}]')
            targ_snrs = rng.uniform(min(targ_snrs), max(targ_snrs), N)
        else:
            print('SNR interval invalid or not given. Defaulting to sampling uniformly from [5,40]')
            targ_snrs = rng.uniform(5, 40, N)
    return targ_snrs


def handle_datatype(datatype) -> tuple[str, Callable]:
    if datatype == 'timeseries':
        return datatype, get_inj_ifos_ts
    elif datatype in ['q-transform', 'q_transform', 'q-trans', 'q_trans']:
        datatype = 'q-transforms'
        from .images import get_inj_ifos_qt
        return datatype, get_inj_ifos_qt
    elif datatype == 'wavelet':
        raise NotImplementedError
    else:
        raise ValueError(f"datatype '{datatype}' not understood. "
                         f"Possibilities are 'timeseries', 'q-transform' and 'wavelet'")


default_jlib_kw = {'n_jobs': 12, 'backend': 'multiprocessing'}


def get_data(targ_snrs=None,
             N=None,
             parallel=False,
             use_tqdm=True,
             prior=None,
             ifos=None,
             noise_ts=None,
             asds=None,
             seed: int | str = 0,
             datatype: str = 'timeseries',
             id_format: Callable = None,
             return_type='default',
             joblib_kwargs=None,
             **inj_func_kwargs) -> tuple[list[np.ndarray], list[np.ndarray]] | list[dict]:
    bilby.utils.logging.disable()

    targ_snrs = handle_snr(targ_snrs, N)

    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()

    bilby.core.utils.random.seed(seed)
    prior_samples = [prior.sample() for _ in range(N)]

    if noise_ts is not None:
        raise NotImplementedError
    #     duration = inj_func_kwargs['waveform_generator'].duration
    #     noises = handle_noise(noise_ts, duration, N)
    # else:
    #     noises = [None for _ in range(targ_snrs)]

    if ifos is None:
        ifos = ["H1", "L1", "V1"]
    # rng = np.random.default_rng(seed)
    datatype, inj_func = handle_datatype(datatype)
    if id_format is None:
        def id_format(i): return f'{seed}.{i:05}'
    if use_tqdm:
        auxfunc = tqdm
    else:
        auxfunc = identity
    if parallel:
        if joblib_kwargs is None:
            joblib_kwargs = {}
        from copy import copy
        _joblib_kwargs = copy(default_jlib_kw)
        _joblib_kwargs.update(joblib_kwargs)
        out = Parallel(**_joblib_kwargs)(
            delayed(inj_func)(snr,
                              prior_s,
                              ifolist=ifos,
                              asds=asds,
                              **inj_func_kwargs)
            for prior_s, snr in auxfunc(list(zip(prior_samples, targ_snrs))))
    else:
        out = [inj_func(snr,
                        prior_s,
                        ifolist=ifos,
                        asds=asds,
                        **inj_func_kwargs)
               for prior_s, snr in zip(prior_samples, targ_snrs)]

    if return_type == 'default':
        return [element[0] for element in out], [element[1] for element in out]
    elif return_type == 'dict':
        return construct_dict(data=out, ifos=ifos, datatype=datatype, id_format=id_format)
