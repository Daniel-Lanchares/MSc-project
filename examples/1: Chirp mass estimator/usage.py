from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.gw import CBCEstimator
from dtempest.gw.sampling import CBCSampleDict

from dtempest.gw.sampling import CBCComparisonSampleDict
# from dtempest.gw.conversion import convert_dataset
from dtempest.gw.catalog import Catalog

# from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.gw.conversions import convert
'''
Chirp mass estimation has been achieved but is yet far from ideal

Its decent on dataset 0 (used to train) and somewhat worse on
dataset 2 (not used to train), but can still be spot on sometimes

Need a more objective way of evaluating performance.
Regardless it is currently at a log_prob of 4, 4.5.
Hopefully can be reduced.
'''

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'Originals'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '1. Chirp mass estimator'
traindir0 = train_dir / 'training_test_0'
traindir1 = train_dir / 'training_test_1'
traindir4 = train_dir / 'training_test_4'
traindir9 = train_dir / 'training_test_9'
catalog_dir = files_dir / 'GWTC-1 Samples'


# dataset = load_rawsets(rawdat_dir, seeds2names(3))

flow = CBCEstimator.load_from_file(traindir4 / 'v0.4.0.pt')
flow0 = CBCEstimator.load_from_file(traindir4 / 'v0.4.2.pt')
flow1 = CBCEstimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow.eval()
# flow.rename('Spv0.9.0')
flow0.eval()
flow1.eval()


from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset

seed = 999
dataset = load_rawsets(rawdat_dir, seeds2names(seed))
dataset.change_parameter_name('d_L', to='luminosity_distance')
testset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')


# for model in [flow]:
#     # model.pprint_metadata(except_keys=['jargon', ('net_config', 'base_net')])
#     sset = model.sample_set(3000, testset, name=model.name)
#     full = sset.full_test()
#     print(model.name)
#     print(full.pp_mean().to_latex(float_format="%.3f"))
#     print('\n\n\n')
# raise Exception

# trainset = convert_dataset(dataset, flow1.param_list)

# Preprocessing can be done manually if preferred, passing preprocess = False to various flow methods
# trainset = flow.preprocess(trainset)

# n = '2.00005'  # '0.00017'
# sdict0 = flow0.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='MKI')
# sdict1 = flow1.sample_dict(5000, trainset['images'][n], reference=trainset['labels'][n], name='v0.1.0')
#
# layout = np.array([['chirp_mass', ], ])
# fig = sdict0.plot_1d_hists(layout, style='bilby', label='shared')
# fig = sdict1.plot_1d_hists(layout, style='deserted', fig=fig, label='shared', same=True,
#                            title=r'$\bf{MKI}$ vs $\bf{v0.1.0}$')
#
# # Interestingly, the first model is more accurate but less precise as opposed to the second in n=4
# error0 = sdict0.accuracy_test(sqrt=True)
# error1 = sdict1.accuracy_test(sqrt=True)
# print(error0)
# print()
# print(error1)
# plt.show()

catalog = Catalog('gwtc-1')
# These 3 may not have been processed correctly, either too low of a mass or bad luck
# del catalog['GW170817'], catalog['GW170809'], catalog['GW170608']

# testset = convert_dataset(catalog.mergers.values(), ['chirp_mass'])
# print(testset['labels']['GW150914'])

#sset0 = flow0.sample_set(50, trainset[:][:], name='v0.1.0')  # TODO: savefile / loadfile methods.
# sset0 = flow0.sample_set(10000, testset, name='gwtc-1 v0.4.0')  # They take some time to make
# sset1 = flow1.sample_set(10000, testset, name='gwtc-1 v0.4.2')  # They take some time to make

# print(sset2['GW170817'].accuracy_test(sqrt=True))  # MSE of ~45 Solar Masses. Will need to look into it
#del sset2['GW170817']

# error0 = sset0.accuracy_test(sqrt=True)
# error1 = sset1.accuracy_test(sqrt=True)
# error2 = sset2.accuracy_test(sqrt=True)


# print(sset0[n])
#print(error0.mean(axis=1))
# print()
# print(error0.mean())
# print()
# print(error1.mean())  # Improves a bit. Will need to take a look at each individually
# print()
# print(trainset['labels'][n])
# layout = np.array([['chirp_mass', ], ])
# fig = sset0[n].plot_1d_hists(layout, style='bilby', label='shared')
# fig = sset1[n].plot_1d_hists(layout, style='deserted', fig=fig, label='shared', same=True,
#                              title=r'$\bf{MKI}$ vs $\bf{v0.1.0}$')
# plt.show()

event = 'GW150914'

#gwtc1 = convert(SampleDict.from_file("https://dcc.ligo.org/public/0157/P1800370/005/GW150914_GWTC-1.hdf5"))

gwtc1 = convert(CBCSampleDict.from_file(catalog_dir/f'{event}_GWTC-1.hdf5'))

image = catalog[event].make_array()
sdict = flow.sample_dict(10000, context=image)
sdict0 = flow0.sample_dict(10000, context=image)
sdict1 = flow1.sample_dict(10000, context=image)

multi = CBCComparisonSampleDict({"GWTC-1": gwtc1,
                                 f"Estimator {flow.name}": sdict,
                                 f"Estimator {flow0.name}": sdict0,
                                 f"Estimator {flow1.name}": sdict1})


fig = multi.plot('chirp_mass', type='hist', kde=True, figsize=(10, 6),
                 legend_kwargs=dict(
    bbox_to_anchor=(0.0, 1.02, 1.0, 0.102), loc=3, handlelength=3, mode="expand",
    borderaxespad=0.0, title=f'Comparison for {event}', title_fontsize='large'
))
# fig = sset2[event].plot(parameter='chirp_mass', type='hist')
# plt.axvline(testset['labels'][event], color='tab:orange')
# plt.title(f'Comparison for {event}', pad=35)
# plt.savefig(f'comparison_{flow1.name}_{event}.png')
plt.gca().set_xlim((10, 50))
# plt.savefig(f'original_models_{event}.png')
plt.show()
# Idea: Model family class: sharing common architecture. Example: Family 0.1: Same config as v0.1.0
'''
MSE improved, need to calculate average uncertainty range (precision_test) (v0.1.0 should be much better).
Interestingly, there seems to be little drop of accuracy between data the model was trained on and not (no overfitting)

The error is still to large, but there v0.1.0 might still be trained over
Training over helps but may require tons of epochs (may switch to test 2: larger dataset + Scheduler)

Variations at 5000 samples are small enough
|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.5509 | M_{O}     |

|            |   MSE from MKIbis | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           17.5734 | M_{O}     |

For Dataset 0 (Part of training)
|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.5572 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.1801 | M_{O}     |

For Dataset 2 (Not part of MKI training)

|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        17.9686 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |            12.239 | M_{O}     |

For Dataset 3 (Not part of neither training)

|            |   MSE from MKI | units     |
|:-----------|---------------:|:----------|
| chirp_mass |        18.4548 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0851 | M_{O}     |
'''


'''
|            |   MSE from v0.1.1 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.9916 | M_{O}     |

|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0809 | M_{O}     |


|            |   MSE from v0.1.0 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           13.0974 | M_{O}     |

|            |   MSE from v0.1.5 | units     |
|:-----------|------------------:|:----------|
| chirp_mass |           12.3885 | M_{O}     |
'''