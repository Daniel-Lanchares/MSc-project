import os
import torch
import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_losses, Pallete, colour_cycler

path = Path("/media/daniel/easystore/Daniel/MSc-files/Examples/4. 7 parameter model")

n = 2
n_models = 3
models = [f'v4.2.{m}' for m in range(n_models)]
resolution = 10  # Carefully with training tests with variable batch_size, could need to load model


def handle_model(model):
    epochs, losses = load_losses(path / f'training_test_{n}', validation=False, model=model)
    epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in epochs.values()])
    loss_avgd = np.array([stage.mean(axis=1) for stage in losses.values()])
    return epochs_avgd, loss_avgd


# vali_epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in vali_epochs.values()])
# valid_avgd = np.array([stage.mean(axis=1) for stage in validations.values()])
# print(epochs[0].max(axis=1))
# print(epochs[1].max(axis=1))
# print(epochs[2].max(axis=1))

epochs_avgd, loss_avgd = zip(*[handle_model(model) for model in models])
epochs_avgd = [epo.reshape((len(epo[0]),)) for epo in epochs_avgd]
loss_avgd = [epo.reshape((len(epo[0]),)) for epo in loss_avgd]
# loss_avgd = np.array(loss_avgd)
print(epochs_avgd)

plt.figure(figsize=(16, 6))

lines = ['-', '--', '-.', ':']

memory = [24.3, 26, 15.7]
times = [0.8558134533961614, 2.221370238860448, 0.6767793861362669]

for i, stage in enumerate(range(n_models)):
    # print(stage)
    timelabel = str(datetime.timedelta(hours=round(times[i], 2)))#.rsplit(':', 1)[0]
    plt.plot(epochs_avgd[i], loss_avgd[i],
             label=f'v4.{n}.{stage} | {memory[i]:.1f} GB of RAM | {timelabel} to train', linestyle=lines[i], color='tab:blue')
    # plt.plot(vali_epochs_avgd[i], valid_avgd[i],
    #          label=f'validation v2.{n}.{stage}', linestyle=lines[i], color='tab:orange')

plt.legend(fontsize=18)
plt.title(f'Loss plot 4.{n}.x\n' +
          r"Extractor net test:  'pre-trained fixed'  vs  'pre-trained overwritten'  vs  'custom untrained'",
          fontsize=18)
plt.xlabel('epochs', fontsize=18)
plt.ylabel('loss score\n(-log_prob)', fontsize=18, labelpad=16.0)
plt.ylim((10, 80))
plt.xticks(range(1, 11), fontsize=16)
plt.yticks(fontsize=16)
# plt.xticks(range(1, len(epochs_avgd.flatten()) + 1))
# plt.savefig('loss_comparison_v0.3.0-v0.7.0.png')
plt.show()
