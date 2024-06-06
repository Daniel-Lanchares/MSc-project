from pathlib import Path
import matplotlib.pyplot as plt

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''
Revisiting old models to get plots and metadata
'''
n = 5
m = 5
letter = ''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3. 6 parameter model'
traindir0 = train_dir / f'training_test_{n}'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / f'v3.{n}.{m}{letter}.pt')
flow0.rename(f'v3.{n}.{m}{letter}')
flow0.eval()
# Old model quirk
flow0.change_parameter_name('d_L', to='luminosity_distance')

# seed = 32
# event = f'{seed}.00020'
seed = 999
event = f'{seed}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
dataset.change_parameter_name('d_L', to='luminosity_distance')
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')
image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(3000, context=image, reference=label)

for model in [flow0]:
    # model.pprint_metadata(except_keys=['jargon', ('net_config', 'base_net')])
    # sset = model.sample_set(3000, trainset[:][:], name=model.name)
    # full = sset.full_test()
    # print(model.name)
    # arr = full.select_parameter('mass_ratio')['accuracy'].to_numpy()
    import numpy as np
    # np.savetxt('tmp2.txt', arr)
    arr = np.loadtxt('tmp2.txt')
    plt.hist(arr, bins=np.logspace(np.log10(0.001), np.log10(130), 80))
    print(np.mean(arr))
    plt.gca().set_xscale("log")
    mean, median, std = np.mean(arr), np.median(arr), np.std(arr)
    plt.axvline(mean, linestyle='-', color='k', label=rf'mean: {mean:.2f}$\pm${std:.2f}')
    plt.axvline(median, linestyle='--', color='k', label=rf'median: {median:.2f}')
    plt.axvspan(mean-std, mean+std, color='tab:purple', alpha=0.3)
    plt.xlabel('Accuracy on mass ratio (q)')
    plt.title(f'Accuracy distribution for model {model.name}')
    plt.legend()
    plt.show()
    # print(full.pp_mean().to_latex(float_format="%.3f"))
    print('\n\n\n')

raise Exception

from scipy import stats
# from pesummary.utils.bounded_1d_kde import bounded_1d_kde
kwargs = {
    'medians': 'all',  # f"Estimator {flow0.name}",
    'hist_bin_factor': 1,
    'bins': 20,
    'title_quantiles': [0.16, 0.5, 0.84],
    'smooth': 1.4,
    'label_kwargs': {'fontsize': 15},
    # 'labelpad': 0.2,
    'title_kwargs': {'fontsize': 15},


    'kde': stats.gaussian_kde,
    'hist_kwargs': {'density': True},
    # 'kde': bounded_1d_kde,
    # 'kde_kwargs': sdict.default_bounds(),
}

fig = plt.figure(figsize=(14, 10))
select_params = flow0.param_list  # ['chirp_mass', 'mass_ratio', 'chi_eff', 'theta_jn', 'luminosity_distance']
fig = sdict.plot(type='corner', parameters=select_params, truths=sdict.select_truths(select_params),
                 fig=fig, **kwargs)
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])
# fig.savefig(f'corner_{flow0.name}_{event}.png', bbox_inches='tight')
plt.show()