from pathlib import Path
import matplotlib.pyplot as plt

import torch

from dtempest.gw import CBCEstimator
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_image

'''

'''
files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3. 6 parameter model'
traindir0 = train_dir / 'training_test_4'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = CBCEstimator.load_from_file(traindir0 / 'overfitting_test.pt')
flow0.change_parameter_name('d_L', to='luminosity_distance')

# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 32#999
event = f'{seed}.00020'
# 32.00009 is a good example of possible degeneration on distance (with inclination maybe?)
# 32.00012 looks great, just needs better declination
# 32.00020 nailed, in spite of messy picture
# 32.00030 nailed, as is fairly high SNR

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
dataset.change_parameter_name('d_L', to='luminosity_distance')
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')

sset0 = flow0.sample_set(3000, trainset[:10][:], name=flow0.name)

error = sset0.accuracy_test(sqrt=True)
#
# sdict = sset0[event]
# fig = sdict.plot(type='corner', truths=trainset['labels'][event])

image = trainset['images'][event]
label = trainset['labels'][event]
sdict = flow0.sample_dict(10000, context=image, reference=label)

fig = sdict.plot(type='corner', truths=trainset['labels'][event])  # TODO: check how to plot only certain parameters
fig = plot_image(image, fig=fig,
                 title_maker=lambda data: f'{event} Q-Transform image\n(RGB = (L1, H1, V1))')
fig.get_axes()[-1].set_position(pos=[0.62, 0.55, 0.38, 0.38])


# For discarding problematic samples
# ras, decs, dists = [], [], []
# for i, dec in enumerate(sdict['dec']):
#     if abs(dec) < 3.14/2:
#         ras.append(sdict['ra'][i])
#         decs.append(dec)
#         dists.append(sdict['d_L'][i])
# sdict['ra'] = np.array(ras)
# sdict['dec'] = np.array(decs)
# sdict['d_L'] = np.array(dists)

# fig = sdict.plot(type='skymap')


print(error.mean(axis=0))
samples = flow0.sample_and_log_prob(3000, trainset['images'][0])
print(-torch.mean(samples[1]))
plt.show()


'''
Dataset 15

|            |   MSE from v3.1.4 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         12.3917   | M_{\odot} |
| mass_ratio |          0.199914 | ø         |
| chi_eff    |          0.193486 | ø         |
| d_L        |        657.537    | Mpc       |
| ra         |          1.50867  | rad       |
| dec        |          0.560015 | rad       |


|            |   MSE from v3.1.6 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         12.4207   | M_{\odot} |
| mass_ratio |          0.199407 | ø         |
| chi_eff    |          0.194147 | ø         |
| d_L        |        658.376    | Mpc       |
| ra         |          1.50471  | rad       |
| dec        |          0.560456 | rad       |

Dataset 30
|            |   MSE from v3.2.1 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         11.6215   | M_{\odot} |
| mass_ratio |          0.200057 | ø         |
| chi_eff    |          0.198596 | ø         |
| d_L        |        625.488    | Mpc       |
| ra         |          1.53152  | rad       |
| dec        |          0.565512 | rad       |

|            |   MSE from v3.2.2 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         11.6114   | M_{\odot} |
| mass_ratio |          0.200534 | ø         |
| chi_eff    |          0.19712  | ø         |
| d_L        |        631.018    | Mpc       |
| ra         |          1.52971  | rad       |
| dec        |          0.564508 | rad       |

Dataset 999

|            |   MSE from v3.3.2 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         12.2341   | M_{\odot} |
| mass_ratio |          0.204443 | ø         |
| chi_eff    |          0.189867 | ø         |
| d_L        |        635.43     | Mpc       |
| ra         |          1.55457  | rad       |
| dec        |          0.558516 | rad       |
tensor(14.3822, grad_fn=<NegBackward0>)

|            |   MSE from v3.3.4 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         12.1986   | M_{\odot} |
| mass_ratio |          0.197778 | ø         |
| chi_eff    |          0.193448 | ø         |
| d_L        |        628.704    | Mpc       |
| ra         |          1.54882  | rad       |
| dec        |          0.561327 | rad       |
tensor(14.2551, grad_fn=<NegBackward0>)

|            |   MSE from v3.3.5 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         11.8113   | M_{\odot} |
| mass_ratio |          0.19896  | ø         |
| chi_eff    |          0.189573 | ø         |
| d_L        |        620.747    | Mpc       |
| ra         |          1.55716  | rad       |
| dec        |          0.555149 | rad       |
tensor(14.2631, grad_fn=<NegBackward0>)

|            |   MSE from v3.4.3 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |         11.8085   | M_{\odot} |
| mass_ratio |          0.636145 | ø         |
| chi_eff    |          0.207094 | ø         |
| d_L        |        645.777    | Mpc       |
| ra         |          1.57683  | rad       |
| dec        |          0.555563 | rad       |
tensor(14.2310, grad_fn=<NegBackward0>)


Dataset 32
|            |   MSE from overfitting_test | units     |
|:-----------|----------------------------:|:----------|
| chirp_mass |                    5.2059   | M_{\odot} |
| mass_ratio |                    0.122846 | ø         |
| chi_eff    |                    0.148944 | ø         |
| d_L        |                  211.735    | Mpc       |
| ra         |                    0.804376 | rad       |
| dec        |                    0.437851 | rad       |
tensor(8.9181, grad_fn=<NegBackward0>)

|            |   MSE from overfitting_test | units     |
|:-----------|----------------------------:|:----------|
| chirp_mass |                    5.20721  | M_{\odot} |
| mass_ratio |                    0.118419 | ø         |
| chi_eff    |                    0.145676 | ø         |
| d_L        |                  203.664    | Mpc       |
| ra         |                    0.776586 | rad       |
| dec        |                    0.433297 | rad       |
tensor(8.1559, grad_fn=<NegBackward0>)

'''