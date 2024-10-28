from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import make_image

files_dir = Path('/mnt/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets' / 'Originals'
trainset_dir = files_dir / 'Trainsets'

seeds = 999
dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
dataset.change_parameter_name('d_L', to='luminosity_distance')
# print(dataset._df.keys())

image = make_image(dict(dataset.T['999.00001']))

fig, axs = plt.subplots(2, 2, figsize=(8, 8))
axf = axs.flatten()

maps = ['Reds', 'Greens', 'Blues']

for i, ax in enumerate(axf):
    ax.set_xticks([])
    ax.set_yticks([])
    if i != 3:
        temp_image = np.zeros_like(image)
        temp_image[:, :, i] = image[:, :, i]
        axf[i].imshow(temp_image)
    else:
        axf[i].imshow(image)
font = 15
axf[0].set_title('LIGO Livingston (L1)', fontsize=font)
axf[1].set_title('LIGO Hanford (H1)', fontsize=font)
axf[2].set_xlabel('VIRGO (V1)', fontsize=font)
axf[3].set_xlabel('Combined (RGB)', fontsize=font)

# plt.arrow(-10, -10, 10, 10, width=10, color='tab:orange')
plt.show()
