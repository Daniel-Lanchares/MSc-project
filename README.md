# CBC_estimator

Implementation of an NPE approach to gravitational wave parameter estimation through the use of normalizing flows.
Once complete it will allow the user to train basic machine learning models to perform a regression task on a CBC signal based on its q-transform

## Usage
The process is meant to have four stages:

- Dataset generation: Creation of 15 parameter injections and its injection in real noise. 
Dataset generation has extra requirements and it is therefore kept separate.
- Dataset preparation: Once the full datasets have been generated they can be viewed and plotted, as well as converted into trainsets that the training pipeline will understand.
- Model training:
- Inference on trained models:

```Python
import numpy as np
import matplotlib.pyplot as plt
from CBC_estimator.dataset.dataset_utils import convert_dataset, plot_hists

dataset = torch.load(dataset_path)

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
layout = np.ones((2,2), dtype=object)
layout[0,0] = ['chi_eff']
layout[1,0] = ['chi_p']
layout[0,1] = ['mass_1', 'mass_2']
layout[1,1] = ['d_L']
fig = plot_hists(dataset, layout, figsize=(10, 8), bins=10)
plt.show()
```
## Requirements
This code relies on glasflow.nflows for its implementation of normalizing flows...