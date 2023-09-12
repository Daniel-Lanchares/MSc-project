# dtempest

The dtempest[¹] library is meant to provide a flexible implementation of an NPE[²] approach to parameter estimation 
through the use of normalizing flows.
Once complete it will allow the user to train basic machine learning models to perform a regression task on an arbitrary 
collection of images.
With the GW package this core functionality is applied to CBC signals based on its q-transform.
Due to dependence on LALSuite algorithms the GW package is restricted to Linux/macOS operating systems

[1]: **D**eep **T**ransform **E**xchangeble **M**odels for **P**osterior **Est**imation.

[2]: **N**eural **P**osterior **E**stimation

## Usage
The process is meant to have four stages:

- Dataset generation: Creation of 15 parameter waveforms and its injection in real noise. 
- Dataset preparation: Once the full datasets have been generated they can be viewed and plotted, as well as converted into trainsets that the training pipeline will understand.
- Model training:
- Inference on trained models:

```Python
import torch
import numpy as np
import matplotlib.pyplot as plt
from dtempest.core.conversion_utils import convert_dataset, plot_hists

dataset = torch.load('path/to/dataset')

# Define the parameters to study the dataset/have your model trained on
params_list = [
    'chirp_mass',
    'chi_eff',
    'd_L',
    'NAP'
]

# Create a trainset to train on this parameters
trainset = convert_dataset(dataset, params_list)

# Or plot the population for another set of parameters by passing a layout
layout = np.ones((2, 2), dtype=object)
layout[0, 0] = ['chi_eff']
layout[1, 0] = ['chi_p']
layout[0, 1] = ['mass_1', 'mass_2']
layout[1, 1] = ['d_L']
fig = plot_hists(dataset, layout, figsize=(10, 8))
plt.show()
```
## Main requirements
This code is built on **PyTorch** and relies on **glasflow.nflows** for its implementation of normalizing flows.

Generating gravitational wave datasets requires various gw related libraries, so one of the **igwn conda environments** is advised 
as an installation starting point. Requirements not yet incorporated to 'setup.py', pycbc required re-installation.