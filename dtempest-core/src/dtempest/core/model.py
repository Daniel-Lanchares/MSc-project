import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import copy
from collections import OrderedDict
from typing import Callable

from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import no_jargon
from .common_utils import identity, check_format
from .flow_utils import create_flow
from .net_utils import create_feature_extractor, create_full_net
from .train_utils import train_model, TrainSet, FeederDataset

from .sampling import SampleSet, SampleDict, MSEDataFrame, MSESeries

class_dict = {
    'sample_set': SampleSet,
    'sample_dict': SampleDict,
    'data_frame': MSEDataFrame,
    'series': MSESeries
}


class Estimator:

    def __init__(self,
                 param_list: list | np.ndarray | torch.Tensor,
                 flow_config: dict,
                 net_config: dict = None,
                 train_config: dict = None,  # Deprecated. For older models only
                 train_history: dict = None,
                 workdir: str | Path = Path(''),
                 name: str = '',
                 mode: str = 'extractor+flow',
                 device: str = 'cpu',
                 preprocess: Callable = identity,
                 jargon: dict = no_jargon
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
        train_config: DEPRECATED
            Configuration dict for training process
        workdir:
            Path to the directory that will hold sample_related files
        mode:
            String to interpret desired architecture ('net+flow', 'extractor (pretrained net)+flow' or just 'flow')
        preprocess:
            preprocess function for the data. It's stored as metadata and can be easily accessed.
            Context is preprocessed under the hood, but can still be done explicitly by the user
            (see preprocess method).
        device:
            Device in which to put the model (cpu/cuda).
        jargon:
            A dict that contains various task-specific info to be defined in each package.
        """

        if train_history is None:
            train_history = OrderedDict()

        self.metadata = {
            'name': name,
            'jargon': jargon,
            'param_list': param_list,

            'net_config': net_config,
            'flow_config': flow_config,
            'train_history': train_history,

            'mode': mode,
            'preprocess': preprocess,
            'device': device
        }

        self.workdir = Path(workdir)
        self.device = device

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

        self.model_to_device(device)

    def __getattr__(self, item: 'str'):
        if item in self.metadata.keys() and item != 'preprocess':
            return self.metadata[item]
        else:
            return self.__dict__[item]

    def model_to_device(self, device):
        """
        Put model to device, and set self.device accordingly.
        """
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Device should be either cpu or cuda, got {device}.")
        self.device = torch.device(device)
        # Commented below so that code runs on first cuda device in the case of multiple.
        # if device == 'cuda' and torch.cuda.device_count() > 1:
        #     print("Using", torch.cuda.device_count(), "GPUs.")
        #     raise NotImplementedError('This needs testing!')
        #     # dim = 0 [512, ...] -> [256, ...], [256, ...] on 2 GPUs
        #     self.model = torch.nn.DataParallel(self.model)
        print(f"Putting posterior model to device {self.device}.")
        self.model.to(self.device)

    @classmethod
    def load_from_file(cls, savefile_path: str | Path, get_metadata: bool = False, **kwargs):
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

        if get_metadata:
            return estimator, metadata
        return estimator

    def save_to_file(self, savefile: str | Path):
        torch.save((self.model.state_dict(), self.metadata), self.workdir / savefile)

    @property
    def name(self):
        return self.metadata['name']

    def rename(self, new_name):
        self.metadata['name'] = new_name

    def change_parameter_name(self, old_name, to):  # to: new name
        self.metadata['param_list'] = [to if name == old_name else name for name in self.metadata['param_list']]

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

    def _append_training_stage(self, train_config):
        n = len(self.metadata['train_history'])
        self.metadata['train_history'].update({f'stage {n}': train_config})
        return n

    def _append_stage_training_time(self, stage_num, train_time):
        self.metadata['train_history'][f'stage {stage_num}'].update({'training_time': train_time})
        print(f'train_time: {train_time} hours')

    def train(self, trainset: TrainSet | str | Path, traindir, train_config, preprocess: bool = True,
              save_loss: bool = True, make_plot: bool = True):

        trainset = check_format(trainset)
        traindir = Path(traindir)
        # Check if traindir exists and make if it does not
        traindir.mkdir(parents=True, exist_ok=True)

        train_config.update({'trainset': trainset.name})
        n = self._append_training_stage(train_config)

        if preprocess:
            trainset = self.preprocess(trainset)

        # FeederDataset is a train-only oriented object meant to work with Pytorch's DataLoader
        trainset = DataLoader(FeederDataset(trainset), batch_size=train_config['batch_size'])

        t1 = time.time()
        epoch, losses = train_model(self.model, trainset, train_config)
        t2 = time.time()
        self._append_stage_training_time(n, (t2 - t1) / 3600)

        zero_pad = 3  # Hard coded for now

        if save_loss:
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            torch.save((epoch, losses), lossdir / 'loss_data.pt')

        if make_plot:
            # Redundancy in lossdir creation to please Pytorch's checker
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            epoch_data_avgd = epoch.reshape(train_config['num_epochs'], -1).mean(axis=1)
            loss_data_avgd = losses.reshape(train_config['num_epochs'], -1).mean(axis=1)

            plt.figure(figsize=(10, 8))
            plt.plot(epoch_data_avgd, loss_data_avgd, 'o--')
            plt.xlabel('Epoch Number')
            plt.ylabel('Log Probability')
            plt.title('Log Probability (avgd per epoch)')
            plt.savefig(lossdir / f'loss_plot_{self.name}_stage_{n}.png', format='png')

    def sample_dict(self,
                    num_samples: int,
                    context: torch.Tensor,
                    params: torch.Tensor = None,
                    name: str = None,
                    reference: torch.Tensor = None,
                    preprocess: bool = True,
                    _class_dict: dict = None):
        samples = self.sample(num_samples, context, preprocess).detach()
        if params is None:
            params = self.param_list
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleDict child object
        sdict_obj = _class_dict['sample_dict']
        sdict = sdict_obj(params, name, jargon=self.jargon)
        for i in range(len(self.param_list)):
            if self.param_list[i] in params:
                sdict[self.param_list[i]] = copy(samples[0, :, i])
                if reference is not None:
                    sdict.truth[self.param_list[i]] = reference[i].item()
        return sdict

    def sample_set(self,
                   num_samples: int,
                   data: TrainSet,
                   params: torch.Tensor = None,
                   name: str = None,
                   preprocess: bool = True,
                   _class_dict: dict = None):
        if params is None:
            params = self.param_list
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleSet child object
        sset_obj = _class_dict['sample_set']
        sset = sset_obj(params, name)
        with tqdm(total=len(data.index), desc=f'Creating SampleSet {name}', ncols=100) as p_bar:
            for event in data.index:
                sset[str(event)] = Estimator.sample_dict(self,
                                                         num_samples,
                                                         data['images'][event],
                                                         params,
                                                         str(event),
                                                         data['labels'][event],
                                                         # If real event: either None or estimation
                                                         preprocess,
                                                         _class_dict)
                p_bar.update(1)
        return sset
