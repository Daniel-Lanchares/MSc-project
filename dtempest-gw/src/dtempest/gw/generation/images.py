# Core modules
import numpy as np

# GW-modules
from gwpy.timeseries.timeseries import TimeSeries as gwpy_TS

# Internal dependencies
from dtempest.gw.generation.generation_utils import prepare_array
from dtempest.gw.generation.timeseries import get_inj_ifos_ts
from dtempest.gw.generation.markIII import generate_timeseries


def ifo_q_transform(tseries: np.ndarray, resol=(128, 128), duration=2, sampling_frequency=1024, **qtrans_kwargs):
    gw_tseries = gwpy_TS(tseries, t0=-duration / 2, sample_rate=sampling_frequency)
    # print(gw_tseries.shape)
    # import matplotlib.pyplot as plt
    # plt.plot(gw_tseries.times, gw_tseries)
    # plt.show()

    qtrans = gw_tseries.q_transform(whiten=False, **qtrans_kwargs)
    image_channel = prepare_array(qtrans.real, resol)
    return np.array(image_channel)


def dataset_q_transform(dataset: list[dict], resol=(128, 128), **qtrans_kwargs):
    for data in dataset:
        data['q-transforms'] = {}
        for ifo in data['timeseries'].keys():
            data['q-transforms'][ifo] = ifo_q_transform(data['timeseries'][ifo], resol, **qtrans_kwargs)
    return dataset  # Technically not needed, right?


def get_inj_ifos_qt(targ_snr,
                    prior_s,
                    sampling_frequency=1024,
                    waveform_generator=None,
                    ifolist=None,
                    noise_ts=None,
                    asds=None,
                    img_res: tuple[int, int] = (128, 128),
                    duration=2,
                    qtrans_kwargs=None):
    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    ts_data, params = get_inj_ifos_ts(targ_snr,
                                      prior_s,
                                      sampling_frequency,
                                      waveform_generator,
                                      ifolist,
                                      noise_ts,
                                      asds)
    qt_data = [ifo_q_transform(tseries, img_res, duration, **qtrans_kwargs) for tseries in ts_data]
    return qt_data, params


def generate_q_transforms(snr_range,
                          prior_s,
                          prior,
                          sampling_frequency=1024,
                          waveform_generator=None,
                          ifolist=None,
                          # noise_ts=None,
                          asds=None,
                          img_res: tuple[int, int] = (128, 128),
                          duration=2,
                          qtrans_kwargs=None):
    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    ts_data, params = generate_timeseries(snr_range,
                                          prior_s,
                                          prior,
                                          sampling_frequency=sampling_frequency,
                                          waveform_generator=waveform_generator,
                                          ifolist=ifolist,
                                          asds=asds)
    qt_data = [ifo_q_transform(tseries, img_res, duration, **qtrans_kwargs) for tseries in ts_data]
    return qt_data, params


def generate_q_transforms_IV(snr_range,
                             prior_s,
                             prior,
                             sampling_frequency=1024,
                             waveform_generator=None,
                             ifolist=None,
                             noise_ts=None,
                             asds=None,
                             img_res: tuple[int, int] = (128, 128),
                             duration=2,
                             qtrans_kwargs=None):
    if qtrans_kwargs is None:
        qtrans_kwargs = {}
    from dtempest.gw.generation.mkIV import generate_timeseries_IV
    ts_data, params = generate_timeseries_IV(snr_range,
                                             prior_s,
                                             prior,
                                             sampling_frequency=sampling_frequency,
                                             waveform_generator=waveform_generator,
                                             ifolist=ifolist,
                                             asds=asds,
                                             noise_ts=noise_ts)
    qt_data = [ifo_q_transform(tseries, img_res, duration, **qtrans_kwargs) for tseries in ts_data]
    return qt_data, params
