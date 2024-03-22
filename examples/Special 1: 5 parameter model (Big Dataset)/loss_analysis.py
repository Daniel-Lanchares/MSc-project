import os
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dtempest.core.common_utils import load_losses, Pallete, colour_cycler

path = Path("/media/daniel/easystore/Daniel/MSc-files/Examples/Special 1. 5 parameter model (Big Dataset)")

n = 5
epochs, losses = torch.load(path / f'training_test_{n}' / 'loss_data_Spv1.5.0_stages_000_to_006' / 'loss_data.pt')
vali_epochs, vali_losses = torch.load(path / f'training_test_{n}' /
                                      'loss_data_Spv1.5.0_stages_000_to_006' / 'validation_data.pt')
# print(epochs.shape)
# print(epochs)
# a, b, c, d = epochs.shape
# print(epochs.reshape((a*b*c, d)))


def process_partition_epochs(arr):
    a, b, c, d = arr.shape
    reshaped = arr.reshape((a * b * c, d))
    for i, epoch in enumerate(reshaped):
        reshaped[i] = epoch + i * np.ones_like(epoch)
    return reshaped.reshape(arr.shape)


def present_by_substages(arr):
    a, b, c, d = arr.shape
    return arr.reshape((a, b * c * d))


epochs = process_partition_epochs(epochs)
vali_epochs = process_partition_epochs(vali_epochs)
# print(present_by_substages(epochs))
# print(present_by_substages(losses))
for i in range(epochs.shape[0]):
    plt.plot(present_by_substages(epochs)[i], present_by_substages(losses)[i], color='tab:blue')

for i in range(vali_epochs.shape[0]):
    plt.plot(present_by_substages(vali_epochs)[i], present_by_substages(vali_losses)[i], color='tab:orange')
plt.ylim((-5, 200))
plt.show()
print(np.min(epochs, axis=-1))
print(np.mean(losses, axis=-1))


# epochs, losses, vali_epochs, validations = load_losses(path / f'training_test_{n}', validation=True)
#
# resolution = 10  # Carefully with training tests with variable batch_size, could need to load model
# epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in epochs.values()])
# loss_avgd = np.array([stage.mean(axis=1) for stage in losses.values()])
# vali_epochs_avgd = np.array([np.around(stage.max(axis=1), 0) for stage in vali_epochs.values()])
# valid_avgd = np.array([stage.mean(axis=1) for stage in validations.values()])
# # print(epochs[0].max(axis=1))
# # print(epochs[1].max(axis=1))
# # print(epochs[2].max(axis=1))
#
# plt.figure(figsize=(12, 6))
#
# lines = ['-', '--', '-.', ':']
#
# for i, stage in enumerate(sorted(epochs)):
#     # print(stage)
#     plt.plot(epochs_avgd[i], loss_avgd[i],
#              label=f'train data v2.{n}.{stage}', linestyle=lines[i], color='tab:blue')
#     plt.plot(vali_epochs_avgd[i], valid_avgd[i],
#              label=f'validation v2.{n}.{stage}', linestyle=lines[i], color='tab:orange')
#
# plt.legend()
# plt.title(f'Loss plot Spv2.{n}.x')
# plt.xlabel('epochs')
# plt.ylabel('loss score\n(-log_prob)')
# plt.ylim((0, 40))
# plt.xticks(range(1, len(epochs_avgd.flatten())+1))
# # plt.savefig('loss_comparison_v0.3.0-v0.7.0.png')
# plt.show()
