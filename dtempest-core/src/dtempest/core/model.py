import re
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from copy import copy
from collections import OrderedDict
from typing import Callable
from pandas._typing import RandomState as pd_RandomState

from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import no_jargon
from .common_utils import identity, check_format
from .flow_utils import create_flow
from .net_utils import create_feature_extractor, create_full_net
from .train_utils import train_model, TrainSet, check_trainset_format, FeederDataset, check_weight_viability

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
                 preprocess: Callable = None,
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

        if preprocess is None:
            preprocess = identity

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

        if 'scales' in flow_config.keys():
            self.scales = self._get_scales(flow_config['scales'])
        else:
            self.scales = torch.ones(len(self.param_list))

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

    def _get_scales(self, scales_config):
        if type(scales_config) is list:
            assert len(scales_config) == len(self.param_list), ('Scales need no be of same length as '
                                                                'list of parameters: '
                                                                f'{len(scales_config)} != {len(self.param_list)}')
            return scales_config
        elif type(scales_config) is dict:
            return torch.tensor(
                [scales_config[param] if param in scales_config.keys() else 1 for param in self.param_list]
            )
        else:
            return torch.ones(len(self.param_list))

    def model_to_device(self, device):
        """
        Put model to device, and set self.device accordingly, adapted from dingo.
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
        samples = self.model.sample(num_samples, context)
        samples[:] = torch.mul(samples[:], self.scales)
        return samples

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
        samples, logprobs = self.model.sample_and_log_prob(num_samples, context)
        samples[:] = torch.mul(samples[:], self.scales)
        return samples, logprobs

    def get_training_stage_seeds(self, stage: int = -1) -> list[int] | None:
        if len(self.metadata['train_history']) == 0:
            return None
        if stage == -1:
            stage = len(self.metadata['train_history']) - 1
        datasets = re.findall(r'\d+', self.metadata['train_history'][f'stage {stage}']['trainset'])
        return [int(data) for data in datasets]

    def turn_net_grad(self, code: str | bool):
        if code == 'on':
            code = True
        elif code == 'off':
            code = False
        elif type(code) is str:
            raise ValueError(f"Argument 'code' = {code} not understood")
        for parameter in self.model._embedding_net.parameters():
            parameter.requires_grad = code

    def dataset_choice(self,
                       seed: int = 0,
                       size: int = 15,
                       data_pool: tuple[int] = (0, 45),
                       new_data_indeces: int | list[int] | list[slice] = None) -> list[int] | np.ndarray:
        # Select the datasets randomly from 0 to 45 and substitute given positions with previously unseen datasets
        previous_seeds = self.get_training_stage_seeds(stage=-1)
        if previous_seeds is None:
            # First time training
            rng = np.random.default_rng(seed)
            return rng.choice(np.arange(*data_pool), size=size, replace=False)
        # TODO Continue, and study case match here.
        if type(new_data_indeces) == int:
            new_data_slices = [slice(new_data_indeces), ]
        elif type(new_data_indeces) == list[int]:
            new_data_slices = [slice(index) for index in new_data_indeces]
        elif type(new_data_indeces) == list[slice]:
            new_data_slices = new_data_indeces
        else:
            raise NotImplementedError(f'Datatype {type(new_data_indeces)} not supported')

        previous_seeds = np.array(previous_seeds)
        for s in new_data_slices:
            new_value = None  # Find value not in previous seeds
            previous_seeds[s] = new_value

    def preprocess(self, trainset: TrainSet | np.ndarray | torch.Tensor):
        if isinstance(trainset, np.ndarray):
            # if it is an array its context from the sampling methods, not an entire set
            trainset = self._preprocess(torch.tensor(trainset))
            trainset = trainset.expand(1, *trainset.shape)
            return trainset

        elif isinstance(trainset, torch.Tensor):
            return trainset  # Has been preprocessed already

        for i in trainset.index:
            trainset.loc[i, 'images'] = self._preprocess(torch.tensor(trainset['images'][i]))
            if len(trainset['images'][i].shape) == 3:
                trainset.loc[i, 'images'] = trainset['images'][i].expand(1, *trainset['images'][i].shape)
            trainset.loc[i, 'labels'] = torch.tensor(trainset['labels'][i])
        return trainset

    def rescale_trainset(self, trainset: TrainSet):
        for event in trainset.index:
            trainset.loc[event, 'labels'] = np.divide(trainset.loc[event, 'labels'], self.scales.numpy())
        return trainset

    def _append_training_stage(self, train_config):
        n = len(self.metadata['train_history'])
        self.metadata['train_history'].update({f'stage {n}': train_config})
        return n

    def _append_stage_training_time(self, stage_num, train_time):
        self.metadata['train_history'][f'stage {stage_num}'].update({'training_time': train_time})
        print(f'train_time: {train_time} hours')

    def _train(self,
               trainset: TrainSet | str | Path,
               traindir: str | Path,
               train_config: dict,
               validation: TrainSet | str | Path = None,
               preprocess: bool = True,
               save_loss: bool = True,
               make_plot: bool = True) \
            -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        # TODO: default train_config attributes

        # Load trainset and rescale (Inner model works with rescaled parameters only)
        trainset = check_trainset_format(trainset)
        trainset = self.rescale_trainset(trainset)

        traindir = Path(traindir)
        # Check if traindir exists and make if it does not
        traindir.mkdir(parents=True, exist_ok=True)

        train_config.update({'trainset': trainset.name})
        n = self._append_training_stage(train_config)

        if preprocess:
            trainset = self.preprocess(trainset)

        # FeederDataset is a train-only oriented object meant to work with Pytorch's DataLoader
        trainset = DataLoader(FeederDataset(trainset), batch_size=train_config['batch_size'])

        if validation is not None:
            validation = self.preprocess(check_format(validation))
            validation = self.rescale_trainset(validation)
            validation = DataLoader(FeederDataset(validation), batch_size=train_config['batch_size'])

        # When training for the first time, check if weights give a reasonable first guess and if not reset them
        if len(self.metadata['train_history']) == 1:
            # Weight check parameters
            if 'weight_check_max_val' in train_config:
                max_val = train_config['weight_check_max_val']
            else:
                max_val = None

            if 'weight_check_max_iter' in train_config:
                max_iter = train_config['weight_check_max_iter']
            else:
                max_iter = None

            if 'weight_check_tenfold' in train_config:
                tenfold = train_config['weight_check_tenfold']
            else:
                tenfold = False
            check_weight_viability(self.model, trainset, max_val, max_iter, tenfold)

        t1 = time.time()
        epochs, losses, vali_epochs, vali_losses = train_model(self.model, trainset, train_config, validation)
        t2 = time.time()
        self._append_stage_training_time(n, (t2 - t1) / 3600)

        zero_pad = 3  # Hard coded for now

        if save_loss:
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            torch.save((epochs, losses), lossdir / 'loss_data.pt')
            if validation is not None:
                torch.save((vali_epochs, vali_losses), lossdir / 'validation_data.pt')

        if make_plot:
            # Redundancy in lossdir creation to please Pytorch's checker
            lossdir = traindir / f'loss_data_{self.name}_stage_{n:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            epoch_data_avgd = epochs.mean(axis=1)
            loss_data_avgd = losses.mean(axis=1)

            plt.figure(figsize=(10, 8))
            plt.plot(epoch_data_avgd, loss_data_avgd, 'o--', label='loss')

            if validation is not None:
                vali_epochs_data_avgd = vali_epochs.mean(axis=1)
                validation_data_avgd = vali_losses.mean(axis=1)
                plt.plot(vali_epochs_data_avgd, validation_data_avgd, 'o--', label='validation')

            plt.xlabel('Epoch Number')
            plt.ylabel('Log Probability')
            plt.title('Log Probability (avgd per epoch)')
            plt.legend()
            plt.savefig(lossdir / f'loss_plot_{self.name}_stage_{n}.png', format='png')

        if validation is not None:
            return epochs, losses, vali_epochs, vali_losses
        else:
            return epochs, losses

    def dataset_partition_training(self,
                                   trainset_paths: list[str | Path] | list[list[Path | str]],
                                   traindir: str | Path,
                                   train_configs: dict | list[dict],
                                   validation: TrainSet | str | Path = None,
                                   preprocess: bool = True,
                                   save_loss: bool = True,
                                   make_plot: bool = True,
                                   n_epochs: int = None,  # TODO: docstring
                                   cutoff: int = -1,
                                   trainset_random_state: pd_RandomState | str = None):
        # Need to introduce a cutoff to have a consistent loss array. Default equivalent to no cutoff condition.
        # String in random_state to signal no randomizing: e.g: 'no_shuffle', but any string will do

        # Partition training consist on loading up a portion of the dataset, train on it for
        # "train_configs[i]["n_epochs"]" epochs and then load the next portion. Ideal for huge datasets.
        if n_epochs is None:
            if isinstance(train_configs, list):
                n_epochs = len(train_configs)
            else:
                raise ValueError("If train_configs is of type 'dict' you need to provide n_epochs")
        if isinstance(train_configs, dict):
            train_configs = [train_configs, ] * n_epochs

        n_trainsets = len(trainset_paths)
        n_subepochs = train_configs[0]['num_epochs']
        n_batches, remainder = divmod(cutoff, train_configs[0]['batch_size'])
        if float(remainder) != 0.0:
            n_batches += 1
        del remainder
        shape = (n_epochs, n_trainsets, n_subepochs, n_batches)

        for config in train_configs:
            if config['num_epochs'] != n_subepochs or config['batch_size'] != train_configs[0]['batch_size']:
                raise ValueError('Dataset partition training requires same number of epochs and batch size for '
                                 f'each TrainSet: You have {n_subepochs} vs {config["num_epochs"]} epochs '
                                 f'and {train_configs[0]["batch_size"]} vs {config["batch_size"]} batch_size.')

        n_pre = len(self.metadata['train_history'])

        full_epochs, full_losses = np.zeros(shape), np.zeros(shape)
        if validation is not None:
            validation = check_format(validation)  # From here onwards it will be a TrainSet if it wasn't already
            vali_batches, remainder = divmod(len(validation), train_configs[0]['batch_size'])
            if float(remainder) != 0.0:
                vali_batches += 1
            del remainder
            vali_shape = (n_epochs, n_trainsets, n_subepochs, vali_batches)
            full_vali_epochs, full_vali_losses = np.zeros(vali_shape), np.zeros(vali_shape)

        for epoch in range(n_epochs):
            print(f'Starting substage {epoch}\n')
            for i, set_path in enumerate(trainset_paths):

                print(f'Loading TrainSet(s) {[Path(path).name for path in set_path]}')
                # Loads and scrambles trainset. Also imposes size = cutoff for all of them
                trainset = TrainSet.load(set_path, name=f'{i + 1} of {len(trainset_paths)}')
                if type(trainset_random_state) is str:
                    n_data = len(trainset)
                    if n_data >= cutoff > 0:
                        condition = trainset.tail(n_data - cutoff).index
                        trainset.drop(condition, inplace=True)
                    elif n_data < cutoff:
                        raise ValueError("Argument 'cutoff' has be greater or equal to TrainSet length"
                                         f'Current values: length {n_data} < cutoff {cutoff}')
                    else:
                        continue  # Dangerous unless you make sure all TrainSets have same length
                else:
                    trainset = trainset.sample(n=cutoff, random_state=trainset_random_state)
                    # print(trainset)

                # TEST ONLY: currently only works for my specific labels
                # d = trainset.index
                # c = d.str.split('.')
                # a = [c[x][-1] for x in range(len(d))]
                # # print(a)
                # b = np.array([int(x) for x in a])
                # condition = trainset[b > cutoff].index

                loss_data = self._train(trainset, traindir, train_configs[epoch], validation, preprocess,
                                        save_loss=False, make_plot=False)
                del trainset
                # print(f'data shape: {loss_data[0].shape}, validation shape: {loss_data[2].shape}')
                full_epochs[epoch, i, :, :] = loss_data[0]  # Maybe list(...) of .flatten() if problematic
                full_losses[epoch, i, :, :] = loss_data[1]
                if validation:
                    full_vali_epochs[epoch, i, :, :] = loss_data[2]
                    full_vali_losses[epoch, i, :, :] = loss_data[3]

        zero_pad = 3  # Hard coded for now
        n_post = len(self.metadata['train_history'])

        if save_loss:
            lossdir = traindir / f'loss_data_{self.name}_stages_{n_pre:0{zero_pad}}_to_{n_post:0{zero_pad}}'
            lossdir.mkdir(parents=True, exist_ok=True)

            torch.save((full_epochs, full_losses), lossdir / 'loss_data.pt')
            if validation is not None:
                torch.save((full_vali_epochs, full_vali_losses), lossdir / 'validation_data.pt')

        if make_plot:
            pass  # We'll see

    def train(self, trainset: TrainSet | str | Path | list[str | Path] | list[list[Path | str]], *args, **kwargs):
        # If the trainset is a list of paths then it will perform a partition training
        if isinstance(trainset, TrainSet | str | Path):
            return self._train(trainset, *args, **kwargs)
        elif isinstance(trainset, list):
            return self.dataset_partition_training(trainset, *args, **kwargs)

    def sample_dict(self,
                    num_samples: int,
                    context: torch.Tensor,
                    params: torch.Tensor = None,
                    name: str = None,
                    reference: torch.Tensor = None,
                    preprocess: bool = True,
                    _class_dict: dict = None) -> SampleDict:
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
                   _class_dict: dict = None) -> SampleSet:
        if params is None:
            params = self.param_list
        if _class_dict is None:
            _class_dict = class_dict

        # Define the corresponding SampleSet child object
        sset_obj = _class_dict['sample_set']
        sset = sset_obj(params, name)
        if hasattr(data, 'name'):
            sset.data_name = data.name
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
