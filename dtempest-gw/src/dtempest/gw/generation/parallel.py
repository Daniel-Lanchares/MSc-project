import os
from multiprocessing import Pool
from collections import OrderedDict
from copy import deepcopy
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm

from gwpy.timeseries import TimeSeries

from pycbc import waveform
from pycbc.detector import Detector
from pycbc.filter import resample_to_delta_t, highpass, matched_filter
from pycbc.psd import interpolate
from bilby.gw.conversion import bilby_to_lalsimulation_spins

from .generation_utils import sine, cosine, gwpy_filter, get_psd, prepare_array, set_ids

default_config = {
    'm_i': (5, 100),

    'a_i': (0, 1),
    'tilt_i': (0, np.pi),
    'phi_jl': (0, 2*np.pi),
    'phi_12': (0, 2*np.pi),

    'ra': [0, 2*np.pi],
    'dec': [-np.pi/2, np.pi/2],
    'd_L': [100, 4000],
    'theta_jn': [0, np.pi],
    'psi': [0, np.pi],

    'phase': [0, 2*np.pi],
    'tGPS': 1187058342,
    'tau': [8, 492],

    'approximant': 'IMRPhenomPv2',
    'minSNR': 5,
    'filt_order': 512,
    'q_interval': (-0.15, 0.1),

    'num_workers': 2,  # 4. Might change in the future. Could be better to pass os.cpu_count()
    'chunksize': 5,
    'seed_zero_pad': 3,
    'log_file': None  # Temporary solution until implementation of logging
}


class Injector:
    def __init__(self, size, seed=0, config=None):
        t1 = perf_counter()

        if config is None:
            config = {}
        self.size, self.seed = size, seed

        self.config = deepcopy(default_config)
        for attr, val in config.items():
            if attr not in self.config.keys():
                raise KeyError(f'Specified key ({attr}) misspelled or not implemented')
            self.config[attr] = val

        # Variable meant to hold noise samples
        self.strains = {}

        self.params = self.generate_base_parameters(self.size, self.seed)
        self.valid_injections = self.parallel_generation(self.params)
        self.acceptance = len(self.valid_injections)/self.size

        t2 = perf_counter()
        self.dt = (t2 - t1) / 60

        st0 = f'\n    Seed {seed:0{config["seed_zero_pad"]}}: '
        st1 = f'Acceptance {len(self.valid_injections):0{2}}/{self.size:0{2}} ({self.acceptance:.2%}) '
        st2 = f'It took {int(self.dt):0{2}} minutes {round((self.dt-int(self.dt))*60):0{2}} seconds'

        message = st0+st1+st2
        print(message)

        if config['log_file'] is not None:
            with open(config['log_file'], 'a') as log_file:
                log_file.write(message)

    def save(self, path):
        torch.save(self.valid_injections, path)
        print('Dataset saved.')

    def generate_base_parameters(self, size, seed=0):

        # Random number generator for reproducibility
        rng = np.random.default_rng(seed=seed)

        # parameters

        # # Component masses (in solar masses)
        m_is = rng.uniform(self.config['m_i'][0], self.config['m_i'][1], [2, size])

        # Spin related
        a_is = rng.uniform(self.config['a_i'][0], self.config['a_i'][1], [2, size])  # Unit-less spin magnitudes
        tilts = sine(self.config['tilt_i'][0], self.config['tilt_i'][1], [2, size], rng=rng)  # Spin tilts
        phi_jls = rng.uniform(self.config['phi_jl'][0],
                              self.config['phi_jl'][1], size)  # Angle between total and orbital angular momentum
        phi_12s = rng.uniform(self.config['phi_12'][0],
                              self.config['phi_12'][1], size)  # Angle between spin projections

        # Localization & Orientation related
        ras = rng.uniform(self.config['ra'][0], self.config['ra'][1], size)  # Right ascension
        decs = cosine(self.config['dec'][0], self.config['dec'][1], size, rng=rng)  # Declination
        dls = rng.uniform(self.config['d_L'][0], self.config['d_L'][1], size)  # Luminosity distance (Mpc)
        theta_jns = sine(self.config['theta_jn'][0],
                         self.config['theta_jn'][1], size, rng=rng)  # Angle between \vec{J} and line of sight
        psis = rng.uniform(self.config['psi'][0], self.config['psi'][1], size)  # Polarization angle

        # Coalescence phase and time
        phases = rng.uniform(self.config['phase'][0], self.config['phase'][1], size)  # Coalescence phase
        tgps = self.config['tGPS']*np.ones(size)  # Reference time (s,2017-08-18 2:25:24 Universal Time)
        taus = rng.uniform(self.config['tau'][0], self.config['tau'][1], size)

        return np.array(list(zip(
            *m_is, *a_is, *tilts, phi_jls, phi_12s, ras, decs, dls, theta_jns, psis, phases, tgps, taus
        )))

    def parallel_generation(self, parameter_lists):
        # Noise strain timeseries
        for ifo in ('L1', 'H1', 'V1'):
            # Read noise from file, notch_filter and resample to 2048 Hz
            strain_pre = TimeSeries.read(f'/home/daniel/Documentos/GitHub/MSc-files/Noise/ts-reference-{ifo}',
                                         format='hdf5')
            strain_pre = gwpy_filter(strain_pre, ifo)

            self.strains[ifo] = resample_to_delta_t(highpass(strain_pre.to_pycbc(), 20.0), 1.0 / 2048).crop(2, 2)

        # with Pool(processes=self.config['num_workers']) as pool:
        #     injections = np.ones(self.size, dtype=object)
        #     with tqdm(total=self.size, desc=f'Injection generation (seed: {self.seed})',
        #               ncols=100) as p_bar:
        #         for i, result in enumerate(pool.imap_unordered(self.generate_injection, parameter_lists,
        #                                                        chunksize=self.config['chunksize'])):
        #             p_bar.update(1)
        #             injections[i] = result  # Could discard None results here already. Test on linux

        # Removed parallel generation, as an update to dependencies made it somehow slower than serialized
        # (used 100% of all of my 12 cores)
        injections = np.ones(self.size, dtype=object)
        with tqdm(total=self.size, desc=f'Injection generation (seed: {self.seed})', ncols=100) as p_bar:
            for i, params in enumerate(parameter_lists):
                result = self.generate_injection(params)
                p_bar.update(1)
                injections[i] = result
        injections = set_ids(injections, self.seed)
        valid_injections = [x for x in injections if x is not None]
        return valid_injections

    def generate_injection(self, parameter_list):

        m1, m2, a1, a2, theta1, theta2, phi_jl, phi_12, \
            ra, dec, dl, theta_jn, psi, phase, tgps, tau = parameter_list

        # Now we convert our angular parameters to cartesian spins
        # 20 is the reference frequency
        result = bilby_to_lalsimulation_spins(theta_jn, phi_jl, theta1, theta2,
                                              phi_12, a1, a2, m1, m2, 20, phase)
        incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = result

        hp, hc = waveform.get_fd_waveform(mass1=m1, mass2=m2,
                                          spin1x=spin_1x, spin2x=spin_2x,
                                          spin1y=spin_1y, spin2y=spin_2y,
                                          spin1z=spin_1z, spin2z=spin_2z,
                                          distance=dl, inclination=incl,
                                          approximant=self.config['approximant'],
                                          coa_phase=phase, f_ref=20,
                                          f_lower=10, delta_f=2 / 15)  # Made to match strain down the line
        # T is the reference time, center of the earth.
        T = tgps + tau  # The rest are calculated with Ifo.time_delay_from_detector(L1)
        q_inter = (T + self.config['q_interval'][0], T + self.config['q_interval'][1])  # Q-transform window

        # I went for OrderedDict to improve print readability
        # (dicts are ordered, but it isn't always the desired order)
        injection = OrderedDict([  # Dictionary including data (q-transforms) and labels (parameters)
            # Not all parameters are expected to be trained on
            ('id', None),  # Training id: seed.zero_pad(valid_id counter), None if unusable for training
            ('gen_id', 'provisional'),
            ('SNR', OrderedDict()),
            ('parameters', OrderedDict([
                ('mass_1', m1),
                ('mass_2', m2),
                ('a_1', a1),
                ('a_2', a2),
                ('tilt_1', theta1),
                ('tilt_2', theta2),
                ('phi_jl', phi_jl),
                ('phi_12', phi_12),
                ('theta_jn', theta_jn),
                ('d_L', dl),
                ('ra', ra),
                ('dec', dec),
                ('psi', psi),
                ('phase', phase),
                ('tc', T),  # Using earth_center as reference

                # This one is somewhat special, allows reconstruction of sky position
                ('NAP', 0)
            ])),
            ('q-transforms', OrderedDict())
        ])

        t = {}  # Times in every interferometer
        for ifo in ('L1', 'H1', 'V1'):
            # Time in this ifo
            t[ifo] = T + Detector(ifo).time_delay_from_earth_center(ra, dec, T)

        for ifo in ('L1', 'H1', 'V1'):

            hpt = hp.to_timeseries()
            hct = hc.to_timeseries()
            h_ref = hpt**2 + hct**2  # Reference for coalescence time

            det = Detector(ifo)
            fp, fc = det.antenna_pattern(ra, dec, psi, t[ifo])
            injection['parameters']['NAP'] += fp ** 2 + fc ** 2

            h_ifo = det.project_wave(hpt, hct, ra, dec, psi)

            # Now we process the signal for injection (whiten for q-transform and non_whiten for SNR)
            strain = self.strains[ifo].time_slice(T - 5, T + 5)
            psd = get_psd(strain)
            w_strain = whiten(strain, psd)

            # First we whiten the reference to get the time to shift the signal by...
            w_h_ref = np.abs(whiten(h_ref.copy(), psd))
            w_h_ref.resize(len(w_strain))
            w_h_ref.start_time = w_strain.start_time
            t_shift_w = w_h_ref.sample_times[np.argmax(w_h_ref)]

            # ... And then we apply it to the signal
            w_h = whiten(h_ifo.copy(), psd)
            w_h.resize(len(w_strain))
            w_h.start_time = w_strain.start_time
            w_h = w_h.cyclic_time_shift(t[ifo] - t_shift_w)
            w_h.start_time = w_strain.start_time

            # We repeat for non_whiting (an empty pass to standardise both series)
            nw_h_ref = np.abs(non_whiten(h_ref.copy(), psd))
            nw_h_ref.resize(len(strain))
            nw_h_ref.start_time = strain.start_time
            t_shift_nw = nw_h_ref.sample_times[np.argmax(nw_h_ref)]

            nw_h = non_whiten(h_ifo.copy(), psd)
            nw_h.resize(len(strain))
            nw_h.start_time = strain.start_time
            nw_h = nw_h.cyclic_time_shift(t[ifo] - t_shift_nw)
            nw_h.start_time = strain.start_time

            # Injections: Normal and whitened
            h_injected = strain.inject(nw_h)
            whiten_inject = w_strain.inject(w_h)

            # Calculate SNR
            template = interpolate(h_ifo.to_frequencyseries(), h_injected.delta_f) * fp  # + hc.copy()*fc
            # Change psd.df to accommodate change in h.df
            psd = interpolate(psd, h_injected.delta_f)
            h_injected_f = h_injected.to_frequencyseries()
            template.resize(len(h_injected_f))

            snr_ifo = matched_filter(template, h_injected_f, psd,
                                     low_frequency_cutoff=20,
                                     high_frequency_cutoff=300).crop(3, 3)
            snr_cut = abs(snr_ifo).time_slice(T - 0.5, T + 0.5)
            snr_arr = np.array(snr_cut)
            indx = snr_arr.argmax()
            injection['SNR'][ifo] = snr_cut[indx]

            ht = TimeSeries.from_pycbc(whiten_inject)
            qtrans = ht.q_transform(outseg=q_inter, whiten=False,
                                    frange=(20, 300), qrange=(4, 64))
            image_channel = prepare_array(qtrans.real)
            injection['q-transforms'][ifo] = np.array(image_channel)

        # Imposition on SNR: At least 1 detector over minimum
        result = None
        for ifo, peak in injection['SNR'].items():
            if peak > self.config['minSNR']:
                injection['id'] = 'valid'
                result = injection

        # print(f'finished injection ? out of {self.size}')
        return result


def whiten(timeseries, psd_series, order=None):
    """
    Whitens a timeseries given a precomputed psd.
    It also resamples and bandpasses
    """
    if order is None:
        order = default_config['filt_order']
    if timeseries.delta_t != 1.0/2048:  # Pycbc's resampling occasionally broke
        timeseries = TimeSeries.from_pycbc(timeseries).resample(2048).to_pycbc()
    f_series = timeseries.to_frequencyseries(delta_f=psd_series.delta_f)
    f_series.resize(len(psd_series))
    whiten_t_series = (f_series / psd_series ** 0.5).to_timeseries()
    whiten_t_series = whiten_t_series.highpass_fir(20, order).lowpass_fir(300, order)
    return whiten_t_series.crop(1, 1)


def non_whiten(timeseries, psd_series, order=None):
    """
    Does not whiten a timeseries given a precomputed psd,
    but adapts it to its standards
    """
    if order is None:
        order = default_config['filt_order']
    if timeseries.delta_t != 1.0/2048:  # Pycbc's resampling occasionally broke
        timeseries = TimeSeries.from_pycbc(timeseries).resample(2048).to_pycbc()
    f_series = timeseries.to_frequencyseries(delta_f=psd_series.delta_f)
    f_series.resize(len(psd_series))
    non_whiten_t_series = f_series.to_timeseries()
    non_whiten_t_series = non_whiten_t_series.highpass_fir(20, order).lowpass_fir(300, order)
    return non_whiten_t_series.crop(1, 1)


def process_strain(strain_pre, ifo, window_inter):
    # Read noise from file, notch_filter and resample to 2048 Hz
    if not isinstance(strain_pre, TimeSeries):
        strain_pre = TimeSeries.from_pycbc(strain_pre)
    strain_pre = gwpy_filter(strain_pre, ifo)

    strain = resample_to_delta_t(highpass(strain_pre.to_pycbc(), 20.0), 1.0 / 2048).crop(2, 2)

    psd = get_psd(strain)
    w_strain = TimeSeries.from_pycbc(whiten(strain, psd))
    #q_inter = (T + self.config['q_interval'][0], T + self.config['q_interval'][1])  # Q-transform window
    qtrans = w_strain.q_transform(outseg=window_inter, whiten=False,
                                  frange=(20, 300), qrange=(4, 64))
    image_channel = prepare_array(qtrans.real)
    return np.array(image_channel)