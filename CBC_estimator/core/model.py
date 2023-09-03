import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import copy
from typing import Callable

from .common_utils import identity, check_format
from .flow_utils import create_flow
from .net_utils import create_feature_extractor, create_full_net
from .sampling import SampleDict
from .train_utils import train_model, TrainSet, QTDataset, DataLoader


class Estimator:  # IDEA: Give the possibility of storing the name(s) of the dataset(s) used to train

    def __init__(self,
                 param_list: list | np.ndarray | torch.Tensor,
                 flow_config: dict,
                 net_config: dict = None,
                 train_config: dict = None,
                 workdir: str | Path = Path(''),
                 mode: str = 'extractor+flow',
                 preprocess: Callable = identity,
                 ):
        """
        Main class of the library. Holds the model and can train and sample

        Parameters
        ====#====
        param_list:
            Iterable of parameters to study
        flow_config:
            Configuration dict for normalizing flow
        net_config:
            Configuration dict for neural network
        train_config:
            Configuration dict for training process
        workdir:
            Path to the directory that will hold sample_related files
        mode:
            String to interpret desired architecture ('net+flow', 'extractor (pretrained net)+flow' or just 'flow')
        preprocess:
            preprocess function for the data. It's stored as metadata and can be easily accessed.
            Context is preprocessed under the hood, but can still be done explicitly by the user
            (see preprocess method).
        """
        self.param_list = param_list
        self.workdir = Path(workdir)

        self.train_config = train_config
        self.net_config = net_config
        self.flow_config = flow_config
        self.mode = mode
        self._preprocess = preprocess
        if mode == 'extractor+flow':
            self.model = create_flow(emb_net=create_feature_extractor(**net_config), **flow_config)
        elif mode == 'net+flow':
            self.model = create_flow(emb_net=create_full_net(**net_config), **flow_config)
        elif mode == 'flow':
            if net_config not in [None, {}]:
                raise ValueError(f'Mode {mode} requires no net: net_config must be None or {dict()}')
            self.model = create_flow(emb_net=None, **flow_config)
        else:
            raise ValueError(f'Either mode {mode} was misspelled or is not implemented')

    @classmethod
    def load_from_file(cls, savefile_path: str | Path, **kwargs):
        """
        savefile should be a torch.save() of a tuple(state_dict, metadata)
        """
        state_dict, metadata = torch.load(savefile_path)
        # At the moment I don't think saving the workdir in the metadata is that useful
        if 'workdir' not in kwargs.keys():
            kwargs['workdir'] = Path('')
        metadata.update(kwargs)
        estimator = cls(**metadata)
        estimator.model.load_state_dict(state_dict)
        return estimator

    def save_to_file(self, savefile: str | Path):
        metadata = {
            'param_list': self.param_list,
            'preprocess': self._preprocess,
            'mode': self.mode,

            'train_config': self.train_config,
            'net_config': self.net_config,
            'flow_config': self.flow_config
        }
        torch.save((self.model.state_dict(), metadata), self.workdir / savefile)

    def eval(self):
        self.model.eval()

    def sample(self, num_samples, context, preprocess: bool = True):
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        return self.model.sample(num_samples, context)

    def log_prob(self, inputs, context, preprocess: bool = True):
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        return self.model.log_prob(inputs, context)

    def sample_and_log_prob(self, num_samples, context, preprocess: bool = True):
        if preprocess:
            context = self.preprocess(context)
        elif len(context.shape) == 3:
            context = context.expand(1, *context.shape)
        return self.model.sample_and_log_prob(num_samples, context)

    def preprocess(self, trainset: TrainSet | np.ndarray):
        if isinstance(trainset, np.ndarray):
            # if it is an array its context from the sampling methods, not an entire set
            trainset = self._preprocess(torch.tensor(trainset))
            trainset = trainset.expand(1, *trainset.shape)
            return trainset

        for i in trainset.index:
            trainset['images'][i] = self._preprocess(torch.tensor(trainset['images'][i]))
            if len(trainset['images'][i].shape) == 3:
                trainset['images'][i] = trainset['images'][i].expand(1, *trainset['images'][i].shape)
            trainset['labels'][i] = torch.tensor(trainset['labels'][i])
        return trainset

    def train(self, trainset: TrainSet | str | Path, traindir, train_config, preprocess: bool = True):
        # TODO: Pretty much everything, this is not how it will work
        #  (will likely feed processed trainset, not dataloader)
        self.train_config = train_config
        trainset = check_format(trainset)
        if preprocess:
            trainset = self.preprocess(trainset)

        train_dataset = QTDataset(trainset)
        dataloader = DataLoader(train_dataset, batch_size=self.train_config['batch_size'])

        epoch, losses = train_model(self.model, dataloader, traindir, self.train_config)

        epoch_data_avgd = epoch.reshape(20, -1).mean(axis=1)
        loss_data_avgd = losses.reshape(20, -1).mean(axis=1)

        plt.figure(figsize=(10, 8))
        plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
        plt.xlabel('Epoch Number')
        plt.ylabel('Mean Squared Error')
        plt.title('Mean Squared Error (avgd per epoch)')
        plt.savefig(traindir / 'loss_plot.png', format='png')

    def sample_dict(self,
                    num_samples: int,
                    context: torch.Tensor,
                    params: torch.Tensor = None,
                    name: str = None,
                    reference: torch.Tensor = None,
                    preprocess: bool = True):
        samples = self.sample(num_samples, context, preprocess).detach()
        if params is None:
            params = self.param_list
        sdict = SampleDict(params, name)
        for i in range(len(self.param_list)):
            if self.param_list[i] in params:
                sdict[self.param_list[i]] = copy(samples[0, :, i])
                if reference is not None:
                    sdict.truth[self.param_list[i]] = reference[i].item()
        return sdict

    def sample_set(self, num_samples, data):
        pass #TODO: Not only this, but also study how to name every image
