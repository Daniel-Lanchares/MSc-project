import os
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_losses

path = Path("/media/daniel/easystore/Daniel/MSc-files/Examples/Special 2. 7 parameter model (Big Dataset)")

n = 1
epochs, losses, validations = load_losses(path / f'training_test_{n}', validation=True)

resolution = 10  # Carefully with training tests with variable batch_size, could need to load model
epochs_avgd = epochs.reshape(resolution, -1).mean(axis=1)
loss_avgd = losses.reshape(resolution, -1).mean(axis=1)
valid_avgd = validations.reshape(resolution, -1).mean(axis=1)

a = 1

plt.plot(epochs_avgd[a:], loss_avgd[a:], label=f'train data v2.{n}.0')
plt.plot(epochs_avgd, valid_avgd, label=f'validation v2.{n}.0')

plt.legend()
plt.title(f'Loss plot Spv2.{n}.0')
plt.xlabel('epochs')
plt.ylabel('loss score\n(-log_prob)')
# plt.savefig('loss_comparison_v0.3.0-v0.7.0.png')
plt.show()
