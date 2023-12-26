import os
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

path = Path("/home/daniel/Documentos/GitHub/MSc-files/Examples/1: Chirp mass estimator")

for i in range(3, 8):
    path_2 = f"training_test_{i}/loss_datav0.{i}.0_stage_0/loss_data.pt"

    epochs, loss = torch.load(path / path_2)

    nptos = int(len(epochs)/10)
    epochs_avgd = epochs.reshape(nptos, -1).mean(axis=1)
    loss_avgd = loss.reshape(nptos, -1).mean(axis=1)

    plt.plot(epochs_avgd, loss_avgd, label=f'v0.{i}.0')

plt.legend()
plt.title('Loss comparison between v0.3.0 & v0.7.0')
plt.xlabel('epochs')
plt.ylabel('loss score\n(-log_prob)')
plt.savefig('loss_comparison_v0.3.0-v0.7.0.png')
plt.show()
