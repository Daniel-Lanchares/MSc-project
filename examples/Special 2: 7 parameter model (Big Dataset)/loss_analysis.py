import os
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_losses, Pallete, colour_cycler

path = Path("/media/daniel/easystore/Daniel/MSc-files/Examples/Special 2. 7 parameter model (Big Dataset)")

n = 6
epochs, losses, vali_epochs, validations = load_losses(path / f'training_test_{n}', validation=True)

resolution = 10  # Carefully with training tests with variable batch_size, could need to load model
epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in epochs.values()])
loss_avgd = np.array([stage.mean(axis=1) for stage in losses.values()])
vali_epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in vali_epochs.values()])
valid_avgd = np.array([stage.mean(axis=1) for stage in validations.values()])
# print(epochs[0].max(axis=1))
# print(epochs[1].max(axis=1))
# print(epochs[2].max(axis=1))

plt.figure(figsize=(12, 6))

lines = ['-', '--', '-.', ':']

for i, stage in enumerate(sorted(epochs)):
    # print(stage)
    plt.plot(epochs_avgd[i], loss_avgd[i],
             label=f'train data v2.{n}.{stage}', linestyle=lines[i], color='tab:blue')
    plt.plot(vali_epochs_avgd[i], valid_avgd[i],
             label=f'validation v2.{n}.{stage}', linestyle=lines[i], color='tab:orange')

plt.legend()
plt.title(f'Loss plot Spv2.{n}.x')
plt.xlabel('epochs')
plt.ylabel('loss score\n(-log_prob)')
plt.ylim((0, 40))
plt.xticks(range(1, len(epochs_avgd.flatten())+1))
# plt.savefig('loss_comparison_v0.3.0-v0.7.0.png')
plt.show()
