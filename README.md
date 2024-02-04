# dtempest

The dtempest[¹] library is meant to provide a flexible implementation of an NPE[²] approach to parameter estimation 
through the use of normalizing flows.
Once in operation it will allow the user to train machine learning models to perform a regression task on an arbitrary 
collection of images.
With the GW package this core functionality is applied to CBC signals based on its q-transform.
Due to dependence on LALSuite algorithms the GW package is restricted to Linux/macOS operating systems.

[1]: **D**eep **T**ransform **E**xchangeable **M**odels for **P**osterior **Est**imation

[2]: **N**eural **P**osterior **E**stimation

This library has been developed as a Master of Science project whose progress can be tracked here (link not yet 
available).

## Usage
The process is meant to have four stages:

- **Dataset generation**: Creation of 15 parameter waveforms and its injection in real noise. 

```Python
from pathlib import Path
from dtempest.gw.generation.parallel import Injector

dataset_path = Path('/path/to/datasets')
seeds = range(10)
zero_pad = 2  # To ensure all numbers are aligned

samples = 2000
for seed in seeds:
    inj = Injector(samples, seed, config={'log_file': dataset_path / 'log.txt',
                                          'seed_zero_pad': zero_pad})
    inj.save(dataset_path / f'Raw_Dataset_{seed:0{zero_pad}}.pt')
```

- **Dataset preparation**: Once the full datasets have been generated they can be viewed and plotted, as well as 
  converted into lighter _TrainSet_ objects that the training pipeline will understand.
```Python
from pathlib import Path
from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset

dataset_path = Path('/path/to/datasets')
trainset_path = Path('/path/to/trainsets')
seeds = range(10)
dataset = load_rawsets(dataset_path, seeds2names(seeds))

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'd_L',
    'ra',
    'dec'
]

# Convert to TrainSet object (pd.DataFrame based instead of OrderedDict based)
# Reusing variable wil optimize RAM if RawSet not necessary
trainset = convert_dataset(dataset, params_list, outpath=trainset_path/'trainset_0_to_10.pkl')

print(trainset) # Right now markdown conversion of table has to be done manually
```
|         | images                                            | labels                                            |
|---------|---------------------------------------------------|---------------------------------------------------|
| 0.00001 | [[[0.09865066, 0.10082455, 0.10322406, 0.10571... | [15.351502954097063, 0.3494616746533443, -0.36... |
| 0.00002 | [[[0.25180227, 0.30129725, 0.3438984, 0.378078... | [66.66450883236487, 0.7015603862088639, -0.076... |
| ...     | ...                                               | ...                                               |
| 9.01145 | [[[0.0058438308, 0.005524259, 0.005327363, 0.0... | [48.48129269856107, 0.6287532813529846, -0.220... |
| 9.01146 | [[[0.026496682, 0.026856517, 0.023931455, 0.01... | [45.425078093766096, 0.9191168696998269, -0.34... |

- **Model training**: Normalizing flows train in much the same way as neural networks do, by optimizing on the 
  parameter space. The log probability of the samples is used as loss to be brought to zero.
```Python
from pathlib import Path
from torchvision import models

from dtempest.core import Estimator
from dtempest.core.common_utils import get_extractor
from dtempest.core.conversion_utils import TrainSet
import dtempest.core.flow_utils as trans

trainset_path = Path('/path/to/trainsets')
traindir = Path('/path/to/train/in')

# Define the parameters to study the dataset/have your model trained on
params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'd_L',
    'ra',
    'dec'
]

model, weights, pre_process = get_extractor('resnet18')

extractor_config = {
    'n_features': 512,
    'base_net': models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    }

flow_config = {  # This config is based on DINGO's transforms
    'input_dim': len(params_list),
    'context_dim': extractor_config['n_features'],
    'num_flow_steps': 5,  # Adding more seemed to help slightly

    'base_transform': trans.mask_affine_autoreg,
    'base_transform_kwargs': {
        'hidden_dim': 2,
        'num_transform_blocks': 3,
        'use_batch_norm': True
    },
    'middle_transform': trans.random_perm_and_lulinear,
    'middle_transform_kwargs': {

    },
    'final_transform': trans.random_perm_and_lulinear,
    'final_transform_kwargs': {

    }
}

flow = Estimator(params_list, flow_config, extractor_config, name='6p_example',
                  workdir=traindir, mode='extractor+flow', preprocess=pre_process)

train_config = {
    'num_epochs': 100,
    'checkpoint_every_x_epochs': 5,  # Not yet implemented
    'batch_size': 128,  
    'optim_type': 'Adam',  # 'SGD'
    'learning_rate': 0.001,
    'grad_clip': None,
    'sched_kwargs': {
        'type': 'StepLR',
        'step_size': 15,  # Note epochs go from 0 to num_epochs-1
        'gamma': 0.75,
        'verbose': True
        }
    }

trainset = TrainSet.load(trainset_path/'trainset_0_to_10.pkl')

flow.train(trainset, traindir, train_config)
flow.save_to_file(traindir/f'{flow.name}.pt')

```
- **Inference on trained models**:

```Python
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from dtempest.core import Estimator
from dtempest.core.conversion_utils import TrainSet

trainset_path = Path('/path/to/trainsets')
traindir = Path('/path/to/have/trained/on')

trainset = TrainSet.load(trainset_path/'trainset_0_to_10.pkl')
event = trainset.index[0]

flow = Estimator.load_from_file(traindir / 'overfitting_test.pt')
flow.eval()

sample_set = flow.sample_set(3000, trainset[:][:], name=flow.name)
sample_dict = sample_set[event]

fig = sample_dict.plot(type='corner', truths=trainset['labels'][event])

error_series = sample_set.accuracy_test(sqrt=True)
print(error_series.mean(axis=0))
samples = flow.sample_and_log_prob(3000, trainset['images'][event])
print(-torch.mean(samples[1]))
plt.show()
```
## Main requirements
This code is built on **PyTorch** and relies on **glasflow.nflows** for its implementation of normalizing flows.

Generating gravitational wave datasets requires various gw related libraries, so one of the **igwn conda 
environments** is advised as an installation starting point. Requirements not yet incorporated to 'setup.py', pycbc 
required re-installation.