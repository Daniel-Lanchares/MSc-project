# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize
from scipy.special import factorial, gamma, hermitenorm
from timm.models.layers import conv2d_same  # Doesn't come with igwn-py310
from tqdm import tqdm
from matplotlib import pyplot as plt
import bilby
from bilby.gw.detector.psd import PowerSpectralDensity
from joblib import Parallel, delayed

from dtempest.core.common_utils import identity

"""
Script provided by Osvaldo Freitas
"""


# Wavelet transform
# https://github.com/tomrunia/PyTorchWavelets/blob/master/wavelets_pytorch/wavelets.py
class Morlet(object):
    def __init__(self, w0=6):
        """w0 is the nondimensional frequency constant. If this is
        set too low then the wavelet does not sample very well: a
        value over 5 should be ok; Terrence and Compo set it to 6.
        """
        self.w0 = w0
        if w0 == 6:
            # value of C_d from TC98
            self.C_d = 0.776

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0, complete=True):
        """
        Complex Morlet wavelet, centred at zero.
        Parameters
        ----------
        t : float
            Time. If s is not specified, this can be used as the
            non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        complete : bool
            Whether to use the complete or the standard version.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given time
        See Also
        --------
        scipy.signal.gausspulse
        Notes
        -----
        The standard version::
            pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
        This commonly used wavelet is often referred to simply as the
        Morlet wavelet.  Note that this simplified version can cause
        admissibility problems at low values of `w`.
        The complete version::
            pi**-0.25 * (exp(1j*w*x) - exp(-0.5*(w**2))) * exp(-0.5*(x**2))
        The complete version of the Morlet wavelet, with a correction
        term to improve admissibility. For `w` greater than 5, the
        correction term is negligible.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        The fundamental frequency of this wavelet in Hz is given
        by ``f = 2*s*w*r / M`` where r is the sampling rate.
        """
        w = self.w0

        x = t / s

        output = np.exp(1j * w * x)

        if complete:
            output -= np.exp(-0.5 * (w ** 2))

        output *= np.exp(-0.5 * (x ** 2)) * np.pi ** (-0.25)

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Morlet"""
        return 4 * np.pi * s / (self.w0 + (2 + self.w0 ** 2) ** 0.5)

    def scale_from_period(self, period):
        """
        Compute the scale from the fourier period.
        Returns the scale
        """
        # Solve 4 * np.pi * scale / (w0 + (2 + w0 ** 2) ** .5)
        #  for s to obtain this formula
        coeff = np.sqrt(self.w0 * self.w0 + 2)
        return (period * (coeff + self.w0)) / (4.0 * np.pi)

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Morlet.
        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Morlet wavelet at the given frequency
        """
        x = w * s
        # Heaviside mock
        Hw = np.array(w)
        Hw[w <= 0] = 0
        Hw[w > 0] = 1
        return np.pi ** -0.25 * Hw * np.exp((-((x - self.w0) ** 2)) / 2)

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.
        This can be worked out analytically by solving
            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** 0.5 * s


class Paul(object):
    def __init__(self, m=4):
        """Initialise a Paul wavelet function of order `m`."""
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Complex Paul wavelet, centred at zero.
        Parameters
        ----------
        t : float
            Time. If `s` is not specified, i.e. set to 1, this can be
            used as the non-dimensional time t/s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Paul wavelet at the given time
        The Paul wavelet is defined (in time) as::
            (2 ** m * i ** m * m!) / (pi * (2 * m)!) \
                    * (1 - i * t / s) ** -(m + 1)
        """
        m = self.m
        x = t / s

        const = (2 ** m * 1j ** m * factorial(m)) / (np.pi * factorial(2 * m)) ** 0.5
        functional_form = (1 - 1j * x) ** -(m + 1)

        output = const * functional_form

        return output

    # Fourier wavelengths
    def fourier_period(self, s):
        """Equivalent Fourier period of Paul"""
        return 4 * np.pi * s / (2 * self.m + 1)

    def scale_from_period(self, period):
        raise NotImplementedError()

    # Frequency representation
    def frequency(self, w, s=1.0):
        """Frequency representation of Paul.
        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the Paul wavelet at the given frequency
        """
        m = self.m
        x = w * s
        # Heaviside mock
        Hw = 0.5 * (np.sign(x) + 1)

        # prefactor
        const = 2 ** m / (m * factorial(2 * m - 1)) ** 0.5

        functional_form = Hw * (x) ** m * np.exp(-x)

        output = const * functional_form

        return output

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.
        This can be worked out analytically by solving
            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return s / 2 ** 0.5


class DOG(object):
    def __init__(self, m=2):
        """Initialise a Derivative of Gaussian wavelet of order `m`."""
        if m == 2:
            # value of C_d from TC98
            self.C_d = 3.541
        elif m == 6:
            self.C_d = 1.966
        else:
            pass
        self.m = m

    def __call__(self, *args, **kwargs):
        return self.time(*args, **kwargs)

    def time(self, t, s=1.0):
        """
        Return a Derivative of Gaussian wavelet,
        When m = 2, this is also known as the "Mexican hat", "Marr"
        or "Ricker" wavelet.
        It models the function::
            ``A d^m/dx^m exp(-x^2 / 2)``,
        where ``A = (-1)^(m+1) / (gamma(m + 1/2))^.5``
        and   ``x = t / s``.
        Note that the energy of the return wavelet is not normalised
        according to `s`.
        Parameters
        ----------
        t : float
            Time. If `s` is not specified, this can be used as the
            non-dimensional time t/s.
        s : scalar
            Width parameter of the wavelet.
        Returns
        -------
        out : float
            Value of the DOG wavelet at the given time
        Notes
        -----
        The derivative of the Gaussian has a polynomial representation:
        from http://en.wikipedia.org/wiki/Gaussian_function:
        "Mathematically, the derivatives of the Gaussian function can be
        represented using Hermite functions. The n-th derivative of the
        Gaussian is the Gaussian function itself multiplied by the n-th
        Hermite polynomial, up to scale."
        http://en.wikipedia.org/wiki/Hermite_polynomial
        Here, we want the 'probabilists' Hermite polynomial (He_n),
        which is computed by scipy.special.hermitenorm
        """
        x = t / s
        m = self.m

        # compute the Hermite polynomial (used to evaluate the
        # derivative of a Gaussian)
        He_n = hermitenorm(m)
        # gamma = scipy.special.gamma

        const = (-1) ** (m + 1) / gamma(m + 0.5) ** 0.5
        function = He_n(x) * np.exp(-(x ** 2) / 2) * np.exp(-1j * x)

        return const * function

    def fourier_period(self, s):
        """Equivalent Fourier period of derivative of Gaussian"""
        return 2 * np.pi * s / (self.m + 0.5) ** 0.5

    def scale_from_period(self, period):
        raise NotImplementedError()

    def frequency(self, w, s=1.0):
        """Frequency representation of derivative of Gaussian.
        Parameters
        ----------
        w : float
            Angular frequency. If `s` is not specified, i.e. set to 1,
            this can be used as the non-dimensional angular
            frequency w * s.
        s : float
            Scaling factor. Default is 1.
        Returns
        -------
        out : complex
            Value of the derivative of Gaussian wavelet at the
            given time
        """
        m = self.m
        x = s * w
        # gamma = scipy.special.gamma
        const = -(1j ** m) / gamma(m + 0.5) ** 0.5
        function = x ** m * np.exp(-(x ** 2) / 2)
        return const * function

    def coi(self, s):
        """The e folding time for the autocorrelation of wavelet
        power at each scale, i.e. the timescale over which an edge
        effect decays by a factor of 1/e^2.
        This can be worked out analytically by solving
            |Y_0(T)|^2 / |Y_0(0)|^2 = 1 / e^2
        """
        return 2 ** 0.5 * s


class Ricker(DOG):
    def __init__(self):
        """The Ricker, aka Marr / Mexican Hat, wavelet is a
        derivative of Gaussian order 2.
        """
        DOG.__init__(self, m=2)
        # value of C_d from TC98
        self.C_d = 3.541


class CWT(nn.Module):
    def __init__(
            self,
            dj=0.0625,
            dt=1 / 2048,
            wavelet=Morlet(),
            # wavelet=DOG(32),
            fmin: int = 20,
            fmax: int = 500,
            output_format="Magnitude",
            trainable=False,
            hop_length: int = 1,
    ):
        super().__init__()
        self.wavelet = wavelet

        self.dt = dt
        self.dj = dj
        self.fmin = fmin
        self.fmax = fmax
        self.output_format = output_format
        self.trainable = trainable  # TODO make kernel a trainable parameter
        self.stride = (1, hop_length)
        # self.padding = 0  # "same"

        self._scale_minimum = self.compute_minimum_scale()

        self.signal_length = None
        self._channels = None

        self._scales = None
        self._kernel = None
        self._kernel_real = None
        self._kernel_imag = None

    def compute_optimal_scales(self):
        """
        Determines the optimal scale distribution (see. Torrence & Combo, Eq. 9-10).
        :return: np.ndarray, collection of scales
        """
        if self.signal_length is None:
            raise ValueError(
                "Please specify signal_length before computing optimal scales."
            )
        J = int(
            (1 / self.dj) * np.log2(self.signal_length * self.dt / self._scale_minimum)
        )
        scales = self._scale_minimum * 2 ** (self.dj * np.arange(0, J + 1))

        # Remove high and low frequencies
        frequencies = np.array([1 / self.wavelet.fourier_period(s) for s in scales])
        if self.fmin:
            frequencies = frequencies[frequencies >= self.fmin]
            scales = scales[0: len(frequencies)]
        if self.fmax:
            frequencies = frequencies[frequencies <= self.fmax]
            scales = scales[len(scales) - len(frequencies): len(scales)]

        return scales

    def compute_minimum_scale(self):
        """
        Choose s0 so that the equivalent Fourier period is 2 * dt.
        See Torrence & Combo Sections 3f and 3h.
        :return: float, minimum scale level
        """
        dt = self.dt

        def func_to_solve(s):
            return self.wavelet.fourier_period(s) - 2 * dt

        return optimize.fsolve(func_to_solve, np.array([1,]))[0]  # Changed 1 -> np.array([1,])

    def _build_filters(self):
        self._filters = []
        for scale_idx, scale in enumerate(self._scales):
            # Number of points needed to capture wavelet
            M = 10 * scale / self.dt
            # Times to use, centred at zero
            t = torch.arange((-M + 1) / 2.0, (M + 1) / 2.0) * self.dt
            if len(t) % 2 == 0:
                t = t[0:-1]  # requires odd filter size
            # Sample wavelet and normalise
            norm = (self.dt / scale) ** 0.5
            filter_ = norm * self.wavelet(t, scale)
            self._filters.append(torch.conj(torch.flip(filter_, [-1])))

        self._pad_filters()

    def _pad_filters(self):
        filter_len = self._filters[-1].shape[0]
        padded_filters = []

        for f in self._filters:
            pad = (filter_len - f.shape[0]) // 2
            padded_filters.append(nn.functional.pad(f, (pad, pad)))

        self._filters = padded_filters

    def _build_wavelet_bank(self):
        """This function builds a 2D wavelet filter using wavelets at different scales

        Returns:
            tensor: Tensor of shape (num_widths, 1, channels, filter_len)
        """
        self._build_filters()
        wavelet_bank = torch.stack(self._filters)
        wavelet_bank = wavelet_bank.view(
            wavelet_bank.shape[0], 1, 1, wavelet_bank.shape[1]
        )
        # See comment by tez6c32
        # https://www.kaggle.com/anjum48/continuous-wavelet-transform-cwt-in-pytorch/comments#1499878
        # wavelet_bank = torch.cat([wavelet_bank] * self.channels, 2)
        return wavelet_bank

    def forward(self, x):
        """Compute CWT arrays from a batch of multi-channel inputs

        Args:
            x (torch.tensor): Tensor of shape (batch_size, channels, time)

        Returns:
            torch.tensor: Tensor of shape (batch_size, channels, widths, time)
        """
        if self.signal_length is None:
            self.signal_length = x.shape[-1]
            self.channels = x.shape[-2]
            self._scales = self.compute_optimal_scales()
            self._kernel = self._build_wavelet_bank()

            if self._kernel.is_complex():
                self._kernel_real = self._kernel.real
                self._kernel_imag = self._kernel.imag

        x = x.unsqueeze(1)

        if self._kernel.is_complex():
            if (
                    x.dtype != self._kernel_real.dtype
                    or x.device != self._kernel_real.device
            ):
                self._kernel_real = self._kernel_real.to(device=x.device, dtype=x.dtype)
                self._kernel_imag = self._kernel_imag.to(device=x.device, dtype=x.dtype)

            # Strides > 1 not yet supported for "same" padding
            # output_real = nn.functional.conv2d(
            #     x, self._kernel_real, padding=self.padding, stride=self.stride
            # )
            # output_imag = nn.functional.conv2d(
            #     x, self._kernel_imag, padding=self.padding, stride=self.stride
            # )
            output_real = conv2d_same(x, self._kernel_real, stride=self.stride)
            output_imag = conv2d_same(x, self._kernel_imag, stride=self.stride)
            output_real = torch.transpose(output_real, 1, 2)
            output_imag = torch.transpose(output_imag, 1, 2)

            if self.output_format == "Magnitude":
                return torch.sqrt(output_real ** 2 + output_imag ** 2)
            else:
                return torch.stack([output_real, output_imag], -1)

        else:
            if x.device != self._kernel.device or x.dtype != self._kernel.dtype:
                self._kernel = self._kernel.to(device=x.device, dtype=x.dtype)

            # output = nn.functional.conv2d(
            #     x, self._kernel, padding=self.padding, stride=self.stride
            # )
            output = conv2d_same(x, self._kernel, stride=self.stride)
            return torch.transpose(output, 1, 2)


def downsample_2d_array_in_one_dimension(array, axis, new_size):
    """Downsamples a 2D numpy array in only one of its dimensions, keeping the average value of the suppressed points.

    Args:
    array: A 2D numpy array.
    axis: The axis to downsample.
    new_size: The new size of the array in the downsampled dimension.

    Returns:
    A downsampled 2D numpy array.
    """

    # Get the original size of the array in the downsampled dimension.
    original_size = array.shape[axis]

    # Calculate the downsampling factor.
    downsampling_factor = original_size // new_size

    # Create a new array to store the downsampled data.
    downsampled_array = np.zeros((array.shape[0], new_size), dtype=array.dtype)

    # Iterate over the downsampled dimension and calculate the average value of the suppressed points.
    for i in range(new_size):
        downsampled_array[:, i] = np.mean(array[:, i * downsampling_factor:(i + 1) * downsampling_factor], axis=1)

    return downsampled_array


def downsample_2d_torch_tensor_in_one_dimension(tensor, axis, new_size):
    """Downsamples a 2D torch tensor in only one of its dimensions, keeping the average value of the suppressed points.

    Args:
    tensor: A 2D torch tensor.
    axis: The axis to downsample.
    new_size: The new size of the array in the downsampled dimension.

    Returns:
    A downsampled 2D torch tensor.
    """

    # Get the original size of the array in the downsampled dimension.
    original_size = tensor.shape[axis]

    # Calculate the downsampling factor.
    downsampling_factor = original_size // new_size

    # Create a new tensor to store the downsampled data.
    downsampled_tensor = torch.zeros((tensor.shape[0], new_size), dtype=tensor.dtype)

    # Iterate over the downsampled dimension and calculate the average value of the suppressed points.
    for i in range(new_size):
        downsampled_tensor[:, i] = torch.mean(tensor[:, i * downsampling_factor:(i + 1) * downsampling_factor], dim=1)

    return downsampled_tensor


# %%
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


# %%
def get_inj_ifos(targ_snr=10,
                 sampling_frequency=1024,
                 waveform_generator=None,
                 prior=None,
                 ifolist=None,
                 asds=None,
                 seed=None):
    """Injects a signal into two interferometers and returns the injected strain data.

    Args:
        targ_snr: The target signal-to-noise ratio (SNR). If targ_snr is equal to -1,
            the function will inject the signal into a noise-free background.
        sampling_frequency: The sampling frequency of the injected signal.
        waveform_generator: The waveform generator to use for the injected signal.

    Returns:
        A tuple of two NumPy arrays, containing the injected strain data of the two
        interferometers.
    """

    assert waveform_generator is not None, 'Cannot inject signals without a waveform generator'
    duration = waveform_generator.duration

    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()
    if ifolist is None:
        ifolist = ["H1", "L1", "V1"]

    # Initialize a counter variable.
    iterator = 0

    # Set up the two interferometers.
    ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)

    # Save the background frequency domain strain data of the two interferometers.
    ifos_bck = [ifo.frequency_domain_strain for ifo in ifos]

    # Sample a set of prior parameters for the signal, including the luminosity distance.
    np.random.seed(seed)
    # TODO: Need to test if this actually works. More attractive option: bilby.core.utils.random.seed(seed)
    prior_s = prior.sample()

    # Set the initial guess for the luminosity distance.
    prior_s['luminosity_distance'] = 1000
    # Kept mostly for debugging purposes
    prior_s['_snr'] = targ_snr

    if targ_snr == -1:
        # Inject the signal into the noise-free background and return the resulting strain data.
        for ifo in ifos:
            ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency,
                                                duration=duration, start_time=duration / 2)
        prior_s['geocent_time'] = 0.
        ifos.inject_signal(prior_s, waveform_generator=waveform_generator)
        return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos]
    else:
        if targ_snr == 0:
            ifos = setup_ifos(duration=duration, sampling_frequency=sampling_frequency, ifos=ifolist, asds=asds)
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
                prior_s['_snr'] = new_snr
                return [np.fft.irfft(ifo.whitened_frequency_domain_strain) for ifo in ifos_], prior_s

            # Adjust the luminosity distance of the signal by the ratio of the current SNR to the target SNR.
            snr_ratios = np.array(getsnr(ifos_) / targ_snr)
            prior_s['luminosity_distance'] = prior_s['luminosity_distance'] * np.prod(snr_ratios)


# %%
# from joblib import Parallel, delayed


def get_data(targ_snrs=None,
             N=None,
             parallel=False,
             use_tqdm=True,
             waveform_generator=None,
             prior=None,
             ifos=None,
             asds=None):
    bilby.utils.logging.disable()
    if targ_snrs is None:
        targ_snrs = []
    if prior is None:
        prior = bilby.gw.prior.BBHPriorDict()
    if ifos is None:
        ifos = ["H1", "L1", "V1"]

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
            targ_snrs = np.random.uniform(min(targ_snrs), max(targ_snrs), N)
        else:
            print('SNR interval invalid or not given. Defaulting to sampling uniformly from [5,40]')
            targ_snrs = np.random.uniform(5, 40, N)
    if use_tqdm:
        auxfunc = tqdm
    else:
        auxfunc = identity
    if parallel:
        out = Parallel(n_jobs=12, backend='multiprocessing')(
            delayed(get_inj_ifos)(snr,
                                  waveform_generator=waveform_generator,
                                  prior=prior,
                                  ifolist=ifos,
                                  asds=asds)
            for snr in auxfunc(targ_snrs))
    else:
        out = [get_inj_ifos(snr,
                            waveform_generator=waveform_generator,
                            prior=prior,
                            ifolist=ifos,
                            asds=asds)
               for snr in auxfunc(targ_snrs)]

    return [element[0] for element in out], [element[1] for element in out]



