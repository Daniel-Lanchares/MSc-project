from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# from pprint import pprint

import torch
# from torchvision import models
from dtempest.core import Estimator


from dtempest.gw.conversion import convert_dataset

'''
Right now this file is useless (outside of testing)
as the 4 parameter model has not yet been achieved
'''


files_dir = Path('/home/daniel/Documentos/GitHub/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'
train_dir = files_dir / 'Examples' / '2: 4 parameter model'
traindir = train_dir / 'training_test_3'

# params_list = [
#     'chirp_mass',
#     'chi_eff',
#     'd_L',
#     'NAP'
#     ]


dataset = []
for seed in range(6, 7):
    dataset = np.concatenate((dataset, torch.load(rawdat_dir/f'Raw_Dataset_{seed}.pt')))
    print(f'Loaded Raw_dataset_{seed}.pt')

flow = Estimator.load_from_file(traindir / 'v2.3.2_NAP.pt')

flow.eval()

trainset = convert_dataset(dataset, flow.param_list, name='Dataset 6')

sset = flow.sample_set(3000, trainset[:][:10], name='v2.3.2_NAP')

error = sset.accuracy_test(sqrt=True)

sdict = sset['6.00001']
fig = sdict.plot(type='corner')
print(error.mean(axis=0))
samples = flow.sample_and_log_prob(1000, trainset['images'][0])
print(-torch.mean(samples[1]))
plt.show()
