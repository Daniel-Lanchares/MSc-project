from pathlib import Path
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

import torch
from torchvision import models
from dtempest.core import Estimator
from dtempest.core.sampling import SampleDict
from dtempest.core.common_utils import load_rawsets, seeds2names

from dtempest.gw.conversion import convert_dataset, plot_images
from dtempest.gw.catalog import Catalog, Merger
import dtempest.core.flow_utils as trans

from pesummary.utils.samples_dict import MultiAnalysisSamplesDict
from pesummary.gw.conversions import convert
'''

'''

files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '3: 6 parameter model'
traindir0 = train_dir / 'training_test_1'
catalog_dir = files_dir / 'GWTC-1 Samples'


flow0 = Estimator.load_from_file(traindir0 / 'v3.1.1.pt')
# flow1 = Estimator.load_from_file(traindir4 / 'v0.4.3.pt')
flow0.eval()
# flow1.eval()

seed = 11
event = f'{seed}.00001'

dataset = load_rawsets(rawdat_dir, seeds2names(seed))
trainset = convert_dataset(dataset, flow0.param_list, name=f'Dataset {seed}')

sset0 = flow0.sample_set(3000, trainset[:][:10], name=flow0.name)

error = sset0.accuracy_test(sqrt=True)

sdict = sset0[event]
fig = sdict.plot(type='corner', truths=trainset['labels'][event])

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
