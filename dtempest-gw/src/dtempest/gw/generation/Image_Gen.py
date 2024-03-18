import os.path

from pycbc import waveform
from pycbc.detector import Detector
from pycbc.filter import resample_to_delta_t, highpass, matched_filter
from pycbc.psd import interpolate, inverse_spectrum_truncation
from bilby.gw.conversion import bilby_to_lalsimulation_spins

from gwpy.timeseries import TimeSeries
from gwpy.signal import filter_design

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from collections import OrderedDict
from skimage.transform import resize
from pathlib import Path
import torch

'''
In this file the raw dataset (list of dictionaries) is created

11 (+NAP) parameters
seed 0: 0588/1000   -> 58.80 %
seed 1: 0588/1000   -> 58.80 %
seed 2: 0583/1000   -> 58.30 %
seed 3: 1159/2000   -> 57.95 %
seed 4: 1216/2000   -> 60.80 %
seed 5: 1166/2000   -> 58.30 %
seed 6: 1196/1000   -> 59.80 %

15 (+NAP) parameters
seed 0: 1043/2000   -> 52.15 %
'''


# The debug list controls which debug plots are made
# ('projected','whiten', 'snr', 'hp', 'psd', 'asd', 'non_whiten', 'q-chanel', 'heighten', 'q-image', 'parameters')
debug = []
heighten_factor = 10
q_interval = (-0.15, 0.1)
seed = 0
size = 2000  # Size of dataset to be generated
savefig = False  # Saves PNG file of q-image if it is in debug
savept = False   # Saves .pt file for PyTorch
show = False    # Shows q-image if it is in debug
checkpoint_every_x_injections = 25  # Has to be a divisor of size

# Consistent color choices for detectors
colors = {'L1': 'tab:blue', 'H1': 'tab:orange', 'V1': 'tab:green', 'K1': 'tab:purple'}

# Order of the filters used
order = 512


def cosine(minimum, maximum, size=1, rng=np.random.default_rng()):
    """
    Adapted from bilby.core.prior.analytical.Cosine and its parent class Prior
    Often used for sampling declination
    """
    val = rng.uniform(0, 1, size)
    norm = 1 / (np.sin(maximum) - np.sin(minimum))
    return np.arcsin(val / norm + np.sin(minimum))


def sine(minimum, maximum, size=1, rng=np.random.default_rng()):
    """
    Adapted from bilby.core.prior.analytical.Sine and its parent class Prior
    Often used for sampling some angular parameters
    """
    val = rng.uniform(0, 1, size)
    norm = 1 / (np.cos(minimum) - np.cos(maximum))
    return np.arccos(np.cos(minimum) - val / norm)


def prepare_array(arr):
    """
    Transform q-transform's real part into a 128 x 128 channel of the image
    """
    resol = 128
    # Might make use of the fact that it is still a Spectrogram before casting it
    arr = np.abs(np.flip(arr, axis=1).T/np.max(arr))
    arr = arr[:, 340:900]
    arr = resize(arr, (resol, resol))
    return arr


def image(inject: dict):
    """
    Creates image array from injection's dictionary
    """
    image_arr = np.dstack((inject['q-transforms']['L1'],
                           inject['q-transforms']['H1'],
                           inject['q-transforms']['V1']))
    return image_arr


def validate(inject: dict):
    """
    Decodes validation into string for better printing
    """
    if inject['id'] is not None:
        return 'valid'
    else:
        return 'invalid'


# def write_pt_file(injects: list):
#     """
#     Dumps injections into .pt file for later load from
#     pytorch.
#
#     Right now uses generation ID but will probably end
#     up using its own ID.
#     """
#     for i, inject in enumerate(injects):
#         # Since not all injections are valid,
#         # redefine id as dataset id
#         inject['gen_id'] = inject['id']
#         inject['id'] = i+1
#     filename = f'Dataset/Raw_dataset.pt'
#
#     torch.save(injects, filename)


def get_psd(timeseries, fftlength=4):
    """
    Creates power spectral density of timeseries
    """
    psd_series = timeseries.psd(fftlength)  # Adapted from a workshop, tinker with parameters
    psd_series = interpolate(psd_series, timeseries.delta_f)
    psd_series = inverse_spectrum_truncation(psd_series, int(fftlength * timeseries.sample_rate),
                                             low_frequency_cutoff=15.0)
    return psd_series


def whiten(timeseries, psd_series):
    """
    Whitens a timeseries given a precomputed psd.
    It also resamples and bandpasses
    """
    if timeseries.delta_t != 1.0/2048:  # Pycbc's resampling occasionally broke
        timeseries = TimeSeries.from_pycbc(timeseries).resample(2048).to_pycbc()
    f_series = timeseries.to_frequencyseries(delta_f=psd_series.delta_f)
    f_series.resize(len(psd_series))
    whiten_t_series = (f_series / psd_series ** 0.5).to_timeseries()
    whiten_t_series = whiten_t_series.highpass_fir(20, order).lowpass_fir(300, order)
    return whiten_t_series.crop(1, 1)


def non_whiten(timeseries, psd_series):
    """
    Does not whiten a timeseries given a precomputed psd,
    but adapts it to its standards
    """
    if timeseries.delta_t != 1.0/2048:  # Pycbc's resampling occasionally broke
        timeseries = TimeSeries.from_pycbc(timeseries).resample(2048).to_pycbc()
    f_series = timeseries.to_frequencyseries(delta_f=psd_series.delta_f)
    f_series.resize(len(psd_series))
    non_whiten_t_series = f_series.to_timeseries()
    non_whiten_t_series = non_whiten_t_series.highpass_fir(20, order).lowpass_fir(300, order)
    return non_whiten_t_series.crop(1, 1)


def gwpy_filter(timeseries: TimeSeries, detector='L1'):
    """
    Notch filters a gwpy TimeSeries
    """
    # bp = filter_design.bandpass(20, 300, timeseries.sample_rate)  # Donne in whiten()
    if detector != 'V1':
        notches = [filter_design.notch(line, timeseries.sample_rate) for line in (60, 120, 240)]
    else:
        notches = [filter_design.notch(line, timeseries.sample_rate) for line in (50, 100, 150)]
    zpk = filter_design.concatenate_zpks(*notches)
    return timeseries.filter(zpk, filtfilt=True)


# Random number generator for reproducibility
rng = np.random.default_rng(seed=seed)


# Parameters:
#       m1, m2:     U(5, 100)M0 (extra: U(5, 35)M0)
#       dL:         U(100, 4000)Mpc
#       iota:       U(0,pi)
#       chi1, chi2: U(-1,1)
# Restriction:      SNR > 5

# 0 and 2 are non-precesing, so I will make a modified script for aligned-spins at some point
# or resort to using just precesing approximants
approximants = ['SEOBNRv4HM_ROM',
                'IMRPhenomPv2',
                'IMRPhenomD']

approx = approximants[1]  # only precessing approximant on the list

m_low, m_high = 5, 100  # Solar Masses
d_low, d_high = 100, 4000  # Mpc
a_low, a_high = 0, 1  # Unit-less

minSNR = 5  # SNR > minSNR

# parameters

# # Component masses (in solar masses)
m_is = rng.uniform(m_low, m_high, [2, size])

# Spin related
a_is = rng.uniform(a_low, a_high, [2, size])    # Unit-less spin magnitudes
tilts = sine(0, np.pi, [2, size], rng=rng)      # Spin tilts
phi_jls = rng.uniform(0, 2*np.pi, size)         # Angle between total and orbital angular momentum
phi_12s = rng.uniform(0, 2*np.pi, size)         # Angle between spin projections

# Localization & Orientation related
ras = rng.uniform(0, 2*np.pi, size)             # Right ascension
decs = cosine(-np.pi/2, np.pi/2, size)          # Declination
dLs = rng.uniform(d_low, d_high, size)          # Luminosity distance (Mpc)
theta_jns = sine(0, np.pi, size, rng=rng)       # Angle between total angular momentum and line of sight
psis = rng.uniform(0, np.pi, size)              # Polarization angle

# Coalescence phase and time
phases = rng.uniform(0, 2*np.pi, size)          # Coalescence phase
tGPS = 1187058342                               # Reference time (s,2017-08-18 2:25:24 Universal Time)
taus = rng.uniform(7, 493, size)                # Time in which to center a noise sample (nth tc = tGPS + nth tau)

valid_id = 0
valid_injections = []
# Noise strain timeseries
strains = {}
for ifo in ('L1', 'H1', 'V1'):
    # Read noise from file, notch_filter and resample to 2048 Hz
    strain_pre = TimeSeries.read(f'ts-reference-{ifo}', format='hdf5')
    strain_pre = gwpy_filter(strain_pre, ifo)

    strains[ifo] = resample_to_delta_t(highpass(strain_pre.to_pycbc(), 20.0), 1.0 / 2048).crop(2, 2)

for i in range(size):
    # First the necessary parameters to create our waveform
    m1, m2 = m_is[:, i]

    a1, a2 = a_is[:, i]
    theta1, theta2 = tilts[:, i]
    phi_jl, phi_12 = phi_jls[i], phi_12s[i]

    dL = dLs[i]
    theta_jn = theta_jns[i]
    phase = phases[i]

    # Now we convert our angular parameters to cartesian spins
    # 20 is the reference frequency
    result = bilby_to_lalsimulation_spins(theta_jn, phi_jl, theta1, theta2,
                                          phi_12, a1, a2, m1, m2, 20, phase)
    incl, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = result

    hp, hc = waveform.get_fd_waveform(mass1=m1, mass2=m2,
                                      spin1x=spin_1x, spin2x=spin_2x,
                                      spin1y=spin_1y, spin2y=spin_2y,
                                      spin1z=spin_1z, spin2z=spin_2z,
                                      distance=dL, inclination=incl,
                                      coa_phase=phase, approximant=approx,
                                      f_ref=20,
                                      f_lower=10, delta_f=2 / 15)  # Made to match strain down the line
    ra = ras[i]
    dec = decs[i]
    psi = psis[i]

    # reference time, that of L1.
    T = tGPS + taus[i]  # The rest are calculated with Ifo.time_delay_from_detector(L1)
    q_inter = (T+q_interval[0], T+q_interval[1]) # Q-transform window

    if 'hp' in debug:
        # print(f'Length of $h_+$: {len(hp)}')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('$h_+$ freq-domain waveform')
        plt.plot(hp.sample_frequencies, hp, color='tab:blue', label=r'$h_+$')
        plt.plot(hp.sample_frequencies, hc, color='tab:orange', label=r'$h_{\times}$')
        plt.xlabel('f (Hz)')
        plt.ylabel('strain')
        plt.legend()

        hpt = hp.to_timeseries()
        hct = hc.to_timeseries()
        plt.subplot(1, 2, 2)
        plt.title(r'$h_+$ and $h_{\times}$ time-domain waveform')
        plt.plot(hpt.sample_times, hpt, color='tab:blue')
        plt.plot(hct.sample_times, hct, color='tab:orange')
        plt.xlabel('t (s)')
        plt.ylabel('strain')
        plt.tight_layout()
        plt.show()

    # I went for OrderedDict to improve print readability
    # (dicts are ordered, but it isn't always the desired order)
    injection = OrderedDict([  # Dictionary including data (q-transforms) and labels (parameters)
        # Not all parameters are expected to be trained on
        ('id', None),  # Training id: seed.zero_pad(valid_id counter), None if unusable for training
        ('gen_id', i + 1),
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
            ('d_L', dL),
            ('ra', ra),
            ('dec', dec),
            ('psi', psi),
            ('phase', phase),
            ('tc', T),  # Using L1 as reference

            # This one is somewhat special, allows reconstruction of sky position
            ('NAP', 0)
        ])),
        ('q-transforms', OrderedDict())
    ])

    heighten = {'q-transforms': {}}  # For debugging purposes
    t = {}  # Times in every interferometer
    L1 = Detector('L1')
    for ifo in ('L1', 'H1', 'V1'):
        Ifo = Detector(ifo)
        # Time in this ifo
        t[ifo] = T + Ifo.time_delay_from_detector(L1, ra, dec, T)

    for ifo in ('L1', 'H1', 'V1'):

        strain_slice = strains[ifo].time_slice(T - 5, T + 5)
        psd = get_psd(strain_slice)
        if 'psd' in debug:
            fs = psd.delta_f * np.arange(psd.data.size)
            plt.loglog(fs, psd, color=colors[ifo])
            plt.xlim(15, 305)
            plt.title('PSD non-whiten strain')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD")
            plt.show()
        if 'asd' in debug:
            fs = psd.delta_f * np.arange(psd.data.size)
            plt.loglog(fs, psd**0.5, color=colors[ifo])
            plt.xlim(15, 305)
            plt.title('ASD non-whiten strain')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("ASD")
            plt.show()

        hpt = hp.to_timeseries()
        hct = hc.to_timeseries()

        Ifo = Detector(ifo)
        fp, fc = Ifo.antenna_pattern(ra, dec, psi, t[ifo])
        injection['parameters']['NAP'] += fp**2 + fc**2
        # Old implementation. To be deleted if project_wave is green lit
        # h_ifo = fp * hpf  # + fc * hcf  # right now geo-centered
        # h_ifo = h_ifo.cyclic_time_shift(h_ifo.start_time + t[ifo] - strain_slice.start_time)
        # dt = Ifo.time_delay_from_earth_center(ra_sample, dec_sample, t[ifo])
        #
        # h_ifo = h_ifo.cyclic_time_shift(dt+0.875)
        # h_ifo.start_time = strain_slice.start_time-0.875
        h_ifo = Ifo.project_wave(hpt, hct, ra, dec, psi)
        # Now we have it projected to our ifo

        if 'projected' in debug:
            h_debug = h_ifo.copy()
            plt.figure(figsize=(8, 5))
            plt.title(r'$h_+$ and $h_{\times}$ ' + f'projected in {ifo}, $f_+$: {fp:.3f}' +
                      r' and $f_{\times}$'+f': {fc:.3f}')
            plt.plot(h_debug.sample_times, h_debug, color=colors[ifo])
            plt.xlabel('t (s)')
            plt.ylabel('strain')
            plt.ylim(- 5e-22, 5e-22)
            plt.show()

        '''
        In this next step I had to shift the signals an (apparently) arbitrary
        amount (-2.5 and -1 for whiten and non_whiten respectively). I am not 
        sure if it is indicative of a mistake or perhaps related to the cropping
        of the various timeseries.
        '''
        # Whiten both with the psd
        whiten_strain = whiten(strain_slice, psd)
        whiten_h = whiten(h_ifo.copy(), psd)
        whiten_h.resize(len(whiten_strain))
        whiten_h = whiten_h.cyclic_time_shift(-2.5 + Ifo.time_delay_from_detector(L1, ra, dec, T))
        whiten_h.start_time = whiten_strain.start_time
        # Old alignment procedure. Erased at a later date
        # Dt = (whiten_strain.sample_times - whiten_h.sample_times)[0]  # Align signal with strain
        # print(Dt)
        # whiten_h = whiten_h.cyclic_time_shift(-Dt)

        # Bring the timeseries to the same standard without whitening
        h = non_whiten(h_ifo.copy(), psd)
        h.resize(len(strain_slice))
        h = h.cyclic_time_shift(-1 + Ifo.time_delay_from_detector(L1, ra, dec, T))
        h.start_time = strain_slice.start_time
        # Old alignment procedure. Erased at a later date
        # Dt2 = (strain_slice.sample_times - h.sample_times)[0]  # Align signal with strain
        # h = h.cyclic_time_shift(-Dt2)

        if 'non_whiten' in debug:
            plt.title(f'$h_+$ and {ifo} noise ('+r'$\tau_0$'+f'= {taus[i]:2.2f} of 500 s)')
            plt.plot(strain_slice.sample_times, strain_slice, color=colors[ifo], label=f'strain {ifo}')
            plt.plot(h.sample_times, h, color='r', label='signal')
            for iifo in ('L1', 'H1', 'V1'):
                plt.plot(t[iifo], 0, 's', label=f'$t_c$ in {iifo}', color=colors[iifo])
            plt.xlabel('t (s)')
            plt.xlim(T - 1, T + 1)
            plt.legend()
            plt.show()

        if 'whiten' in debug:
            plt.title(f'Whitened $h_+$ and {ifo} noise ('+r'$\tau_0$'+f'= {taus[i]:2.2f} of 500 s)')
            plt.plot(whiten_strain.sample_times, whiten_strain, color=colors[ifo], label=f'strain {ifo}')
            plt.plot(whiten_h.sample_times, whiten_h, color='r', label='signal')
            for iifo in ('L1', 'H1', 'V1'):
                plt.plot(t[iifo], 0, 's', label=f'$t_c$ in {iifo}', color=colors[iifo])
            plt.xlabel('t (s)')
            plt.xlim(T - 1, T + 1)
            plt.legend()
            plt.show()

        # Injections: Normal and whitened
        h_injected = strain_slice.inject(h)
        whiten_inject = whiten_strain.inject(whiten_h)

        # Calculate SNR
        template = interpolate(hp.copy(), h_injected.delta_f)*fp  # + hc.copy()*fc
        # Change psd.df to accommodate change in h.df
        psd = interpolate(psd, h_injected.delta_f)
        h_injected_f = h_injected.to_frequencyseries()
        template.resize(len(h_injected_f))

        snr_ifo = matched_filter(template, h_injected_f, psd,
                                 low_frequency_cutoff=20,
                                 high_frequency_cutoff=300).crop(3, 3)
        snr_cut = abs(snr_ifo).time_slice(T-0.5, T+0.5)
        snr_arr = np.array(snr_cut)
        indx = snr_arr.argmax()
        injection['SNR'][ifo] = snr_cut[indx]

        if 'snr' in debug:
            # Right now SNR is alright, perhaps lower than expected
            # (most injections don't pass the test)
            print(f'Max SNR in {ifo}: {snr_cut[indx]}')
            plt.title(f'SNR in {ifo} for injected signal')
            plt.xlabel('t (s)')
            plt.ylabel('SNR')
            plt.plot(snr_ifo.sample_times, abs(snr_ifo), color=colors[ifo])
            plt.plot(snr_cut.sample_times[indx], snr_cut[indx], 'r*', label='max SNR\naround $t_c$')
            for iifo in ('L1', 'H1', 'V1'):
                plt.plot(t[iifo], 0, 's', label=f'$t_c$ in {iifo}', color=colors[iifo])
            plt.legend(loc='upper right')
            plt.xlim(T-2, T+2)
            plt.show()

        ht = TimeSeries.from_pycbc(whiten_inject)
        # ht = TimeSeries.from_pycbc(h_injected)
        qtrans = ht.q_transform(outseg=q_inter, whiten=False,
                                frange=(20, 300), qrange=(4, 64))
        image_channel = prepare_array(qtrans.real)
        injection['q-transforms'][ifo] = np.array(image_channel)

        if 'q-channel' in debug:
            plt.imshow(np.abs(np.flip(qtrans.real, axis=1)/np.max(qtrans.real)))
            plt.show()

        if 'heighten' in debug:

            # heighten signal to better locate injections buried in the noise
            heighten_inject = whiten_strain.inject(whiten_h * heighten_factor)
            ht = TimeSeries.from_pycbc(heighten_inject)
            qtrans = ht.q_transform(outseg=q_inter, whiten=False,
                                    frange=(20, 300), qrange=(4, 64))
            image_channel = prepare_array(qtrans.real)
            heighten['q-transforms'][ifo] = np.array(image_channel)

    if 'parameters' in debug:
        print(f'injection number {i + 1}\'s parameters')
        pprint(dict(injection['parameters'], order=True))

    if 'snr' in debug:
        NAP = injection['parameters']['NAP']
        print(f'Network Antenna Pattern: {NAP}')
        print('SNR peaks')
        pprint(injection['SNR'])

    # Imposition on SNR: At least 1 detector over minimum
    for ifo, peak in injection['SNR'].items():
        if peak > minSNR:
            injection['id'] = 'valid'
    if injection['id'] == 'valid':
        # Give the injection a valid id and append it to the cache list
        valid_id += 1
        injection['id'] = f'{seed}.{valid_id:05}'
        # print(int(injection['id'].split('.')[1]))
        valid_injections.append(injection)

    if 'q-image' in debug:
        validation = validate(injection) # String to indicate valid/invalid injection
        if 'heighten' in debug:
            plt.figure(figsize=(8, 5))
            plt.subplot(1, 2, 1)
            plt.title(f'Heighten Q-Transform\nwith H = {heighten_factor}·h')
            plt.imshow(image(heighten))
            plt.subplot(1, 2, 2)
            plt.tight_layout()
        plt.title(f'Images/injection_{i+1}({validation}).png\nQ-Transform image (RGB = (L1, H1, V1))')
        plt.imshow(image(injection))
        if savefig:
            plt.savefig(Path(f'Images/injection_{i+1}({validation}).png'), format='png')
        if show:
            plt.show()

    print(f'Finished injection {i+1} out of {size}')
    if (i+1) % checkpoint_every_x_injections == 0 and savept:
        if not os.path.exists('Dataset/Checkpoint.pt'):
            torch.save([], 'Dataset/Checkpoint.pt')
        previous_inj = torch.load('Dataset/Checkpoint.pt')
        injections = np.concatenate((previous_inj, valid_injections))
        if i+1 != size:
            torch.save(injections, 'Dataset/Checkpoint.pt')
        else:
            torch.save(injections, f'Dataset/Raw_Dataset_{seed}.pt')
            os.remove('Dataset/Checkpoint.pt')
        valid_injections = []
        print('Checkpoint reached')


print(f'\nRecuperation coefficient: {valid_id/size}')
