from pathlib import Path

# import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import TABLEAU_COLORS

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict, CBCComparisonSampleDict

from dtempest.gw.conversion import plot_image
from dtempest.gw.catalog import Merger

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
from pesummary.gw.conversions import convert


def basic_corner():

    cat = 'gwtc-1'
    event = 'GW150914'
    # event = 'GW170823'
    # event = 'GW170104'
    # event = 'GW170818'

    # cat = 'gwtc-2.1'
    # event = 'GW190814_211039' # .split('_')[0] # GW190814 is a bit special...
    # event = 'GW190503_185404'
    # event = 'GW190517_055101'

    # cat = 'gwtc-3'
    # event = 'GW200129_065458'
    # event = 'GW200208_222617'
    # event = 'GW200224_222234'
    # event = 'GW200220_061928'  # Not as good, as expected. Way outside learning prior
    # event = 'GW200308_173609'


    # Same as in generation (Should attach then as model metadata, never got round to it)
    resol = (50, 200) #(128, 128)  # (32, 48)  # (48, 72)
    img_window = (-0.3, 0.1) # (-0.065, 0.075)

    from_internet = True
    if from_internet:
        # Needs the specific address for each event, but it is enough for a quick test
        gwtc = convert(CBCSampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))
    else:
        if cat == 'gwtc-1':
            gwtc = convert(CBCSampleDict.from_file(catalog_1 / f'{event}_GWTC-1.hdf5'))
        elif cat == 'gwtc-2.1':
            gwtc = convert(CBCSampleDict.from_file(catalog_2 / f'{event}_cosmo.h5')['C01:Mixed'])
        elif cat == 'gwtc-3':
            gwtc = convert(CBCSampleDict.from_file(catalog_3 / f'{event}_cosmo.h5')['C01:Mixed'])
        else:
            gwtc = None

    merger = Merger(event, cat, img_res=resol, image_window=img_window)


    if 'geocent_time' in gwtc.keys() and 'geocent_time' in flow0.param_list:
        from pesummary.utils.array import Array
        # gwtc['geocent_time'] = Array(gwtc['geocent_time'].to_numpy() % 24*3600)
        gwtc['geocent_time'] = Array((gwtc['geocent_time'].to_numpy() % 24 * 3600)/(24*3600))
        flow0.scales[flow0.param_list.index('geocent_time')] = 1

    image = merger.make_array()
    sdict = flow0.sample_dict(10000, context=image)

    multi = CBCComparisonSampleDict({cat.upper(): gwtc,
                                     f"Estimator {flow0.name}": sdict,})
                                     # f"Estimator {flow2.name}": sdict2})  # More comparisons are possible

    del sdict, gwtc

    # fig = plt.figure(figsize=(12, 10))
    select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']

    kwargs = {
        'medians': 'all',
        'hist_bin_factor': 1,
        'bins': 20,
        'title_quantiles': [0.16, 0.5, 0.84],
        'smooth': 1.4,
        'label_kwargs': {'fontsize': 25},  # 25 for GWTC-1, 25 for GWTC-2/3?
        # 'labelpad': 0.2,
        'title_kwargs': {'fontsize': 20},  # 20 for GWTC-1, 20 for GWTC-2/3?

        'kde': stats.gaussian_kde
        # 'kde': bounded_1d_kde,
        # 'kde_kwargs': multi.default_bounds(),
    }

    fig = multi.plot(type='corner', parameters=select_params, **kwargs)
    del multi
    plt.tight_layout(h_pad=-4.5, w_pad=-0.8)  # h_pad -1 for 1 line title, -3 for 2 lines
    # fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
    #                  smooth=smooth, smooth1d=smooth, medians=True, fig=fig)
    fig = plot_image(image, fig=fig,
                     title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))',
                     title_kwargs={'fontsize': 40},  # 40 for GWTC-1, 40 for GWTC-2/3?
                     aspect=resol[1] / resol[0])
    fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])

    from dtempest.core.common_utils import redraw_legend
    redraw_legend(fig,
                  fontsize=25,  # 25 for GWTC-1, 30 for GWTC-2/3 approx
                  loc='upper center',
                  bbox_to_anchor=(0.4, 0.98),
                  handlelength=2,
                  linewidth=5)

    # To remove gridlines
    for ax in fig.get_axes():
        ax.grid(False)

    # fig.savefig(f'{event}_{flow0.name}_corner.pdf', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    '''
    Sampling from trained model and comparison to GWTC data
    
    In my end the matplotlib style is more latex-like, for some reason. 
    '''
    name = 'PlaceHolder' # Model name
    files_dir = Path('')
    # rawdat_dir = files_dir / 'Raw Datasets'
    trainset_dir = files_dir / 'Trainsets'
    train_dir = files_dir / 'Model'
    outdir = train_dir / 'training_test_1' # This one is created automatically, the rest are not
    catalog_1 = files_dir / 'GWTC-1 Samples'
    catalog_2 = files_dir / 'GWTC-2.1 Samples'
    catalog_3 = files_dir / 'GWTC-3 Samples'

    flow0 = CBCEstimator.load_from_file(outdir / f'{name}.pt')
    flow0.eval()

    print(f'{flow0.name} metadata')
    flow0.pprint_metadata()

    basic_corner()
