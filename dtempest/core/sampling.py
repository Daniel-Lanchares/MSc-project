from collections import OrderedDict, namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
import torch
from tqdm import tqdm

from .common_utils import PrintStyle
from .conversion_utils import get_param_alias, get_param_units
from .train_utils import TrainSet

# class SampleArray(np.ndarray):
#     # read this, might be useful
#     # https://numpy.org/doc/stable/user/basics.subclassing.html
#     def

average_dict = {
    'median': torch.median,
    'mean': torch.mean
}


class SampleDict(OrderedDict):

    def __init__(self, parameters, name: str = None, average: str = 'median'):
        super().__init__()
        self._truth = OrderedDict()
        self.parameters = parameters
        if name is None:
            self.name = type(self).__name__
        else:
            self.name = name
        self.average = average_dict[average]

    @property
    def truth(self):
        return self._truth

    @truth.setter
    def truth(self, new_truth):
        if len(self.truth) != 0:
            print(f'This {type(self).__name__} already has truth values')
            pprint(self.truth)
            print('Are you sure you want to overwrite some / all of them? [y/n]')
            if input() not in ['y', 'yes']:
                print('Aborting operation then')
                return
        if isinstance(new_truth, dict):
            self._truth.update(new_truth)
        elif isinstance(new_truth, torch.Tensor):
            for i, param in enumerate(self.parameters):
                try:
                    self._truth[param] = new_truth[i].item()
                except IndexError:
                    self._truth[param] = new_truth.item()
        elif isinstance(new_truth, list | np.ndarray):
            for i, param in enumerate(self.parameters):
                self._truth[param] = new_truth[i]

    def get_average(self):
        return torch.Tensor([self.average(self[param]) for param in self.parameters])

    def plot_1d_hists(self, param_array, fig=None, figsize=None, quantiles=(0.16, 0.84),
                      average=False, same=False, **kwargs):
        if fig is not None and not same:
            print(f'{PrintStyle.red}Warning: Providing a figure to plot_1d_hists writes over its previous contents '
                  f'unless {PrintStyle.lightred}same {PrintStyle.reset}={PrintStyle.orange} True{PrintStyle.red} '
                  f'is specified, in which case histograms are presented together.{PrintStyle.reset}')
            fig.clear()

        if fig is None:
            fig = plt.figure(figsize=figsize)
        if same and 'label' not in kwargs:
            kwargs['label'] = self.name

        if 'title' not in kwargs:
            fig.suptitle(f'{self.name} histograms')
        else:
            fig.suptitle(kwargs['title'])
            del kwargs['title']

        layout = param_array.shape
        flat_array = param_array.flatten()
        for i in range(len(flat_array)):
            parameter = flat_array[i]
            fig = self.plot_1d_hist(parameter, fig=fig, figsize=figsize, quantiles=quantiles,
                                    plot_layout=(*layout, i + 1), average=average, same=same,
                                    **kwargs)
        plt.tight_layout()
        return fig

    def plot_1d_hist(self, parameter, fig: plt.Figure = None, figsize=None, quantiles=(0.16, 0.84),
                     plot_layout=(1, 1, 1), average=False, style: str = 'bilby', same=False,
                     **kwargs):
        """
        Plots a histogram of a given parameter on a single subplot

        Parameters
        ----------
        parameter : list.
            Parameter to plot.
        fig : matplotlib.pyplot.figure, optional
            Matplotlib.pyplot figure to be plotted on. Especially useful to paint
            various plots manually. The default is None.
        figsize : tuple, optional.
            'figsize' parameter for fig (ignored if fig is given). The default is None.
        plot_layout : tuple, optional
            Plot layout. Useful mostly to paint
            various plots manually. The default is (1,1,1).
        average: bool, optional
            Whether to show the average of the data (either median or mean)
            The default is False
        style: str, optional
            Choosing a predefined style is ideal for plotting multiple SampleDicts
            in the same axis. The default is 'bilby' (as in the bilby library).
        same: bool, optional
            If True it will plot the histogram in the corresponding preexisting
            axis of the given figure (provided that both layouts match).
        **kwargs : dict.
            Keyword arguments to set style and to be passed to plt.hist().

        Returns
        -------
        fig : matplotlib.pyplot.figure
            updated figure with the histogram now plotted.

        """
        default_kwargs = set_hist_style(style)
        default_kwargs.update(kwargs)
        kwargs = default_kwargs
        hist_kwargs = deepcopy(kwargs)
        for key in ['truth_color', 'truth_format',
                    'average_color', 'average_format',
                    'errorbar_color', 'errorbar_format']:
            del hist_kwargs[key]

        data = self[parameter]

        if fig is not None:
            # Whether to plot in existing axis or create new one.
            if same:
                ax = np.array(fig.get_axes())[plot_layout[-1]-1]
            else:
                ax = fig.add_subplot(*plot_layout)
        else:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(*plot_layout)

        summary = self.get_one_dimensional_median_and_error_bar(parameter, quantiles=quantiles)

        name = get_param_alias(parameter)
        title = (f'{name}: {summary.median:.2f}'
                 f'$^{{+{summary.plus:.2f}}}_{{-{summary.minus:.2f}}}$ ({get_param_units(parameter)})')
        if 'label' in hist_kwargs and hist_kwargs['label'] == 'shared':
            title = f'{name}'
            hist_kwargs['label'] = (f'{self.name}: {summary.median:.2f}'
                                    f'$^{{+{summary.plus:.2f}}}_{{-{summary.minus:.2f}}}$ '
                                    f'({get_param_units(parameter)})')
        ax.hist(data, **hist_kwargs)

        if average:
            ax.axvline(self.average(data), ls=kwargs['average_format'], color=kwargs['average_color'])
        ax.axvline(summary.median - summary.minus, ls=kwargs['errorbar_format'], color=kwargs['errorbar_color'])
        ax.axvline(summary.median + summary.plus, ls=kwargs['errorbar_format'], color=kwargs['errorbar_color'])
        if parameter in self.truth.keys():
            ax.axvline(self.truth[parameter], ls=kwargs['truth_format'], color=kwargs['truth_color'])

        ax.set_xlabel(f'{get_param_alias(parameter)} ({get_param_units(parameter)})')

        ax.set_title(title)
        if 'label' in hist_kwargs:
            ax.legend()
        return fig

    def get_one_dimensional_median_and_error_bar(self, key, fmt='.2f',
                                                 quantiles=(0.16, 0.84)):
        # Credit: https://git.ligo.org/lscsoft/bilby/-/blob/master/bilby/core/result.py
        """ Calculate the median and error bar for a given key

        Parameters
        ==========
        key: str
            The parameter key for which to calculate the median and error bar
        fmt: str, ('.2f')
            A format string
        quantiles: list, tuple
            A length-2 tuple of the lower and upper-quantiles to calculate
            the errors bars for.

        Returns
        =======
        summary: namedtuple
            An object with attributes, median, lower, upper and string

        """
        summary = namedtuple('summary', ['median', 'lower', 'upper', 'string'])

        if len(quantiles) != 2:
            raise ValueError("quantiles must be of length 2")

        quants_to_compute = np.array([quantiles[0], 0.5, quantiles[1]])
        quants = np.percentile(self[key], quants_to_compute * 100)
        summary.median = quants[1]
        summary.plus = quants[2] - summary.median
        summary.minus = summary.median - quants[0]

        fmt = "{{0:{0}}}".format(fmt).format
        string_template = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        summary.string = string_template.format(
            fmt(summary.median), fmt(summary.minus), fmt(summary.plus))
        return summary

    def accuracy_test(self, sqrt: bool = False):
        if len(self.truth) == 0:
            print(f'{PrintStyle.red}{type(self).__name__} has no truth dictionary (reference values) to compare to.')
            print(f'Accuracy test aborted.{PrintStyle.reset}')
            return
        error = MSESeries(index=self.parameters, name=f'MSE from {self.name}', sqrt=sqrt)
        for param in self.parameters:
            avg = self.average(self[param]).item()
            ref = self.truth[param]
            if sqrt:
                error[param] = np.abs(avg - ref)
            else:
                error[param] = (avg-ref)**2
        return error


class SampleSet(SampleDict):
    def accuracy_test(self, sqrt: bool = False):
        # data = {event: sdict.accuracy_test(sqrt) for event, sdict in self.items()}
        data = OrderedDict()
        with tqdm(total=len(self.keys()), desc=f'Running accuracy test for {self.name}', ncols=100) as p_bar:
            for event, sdict in self.items():
                data[event] = sdict.accuracy_test(sqrt)
                p_bar.update(1)

        return MSEDataFrame(data=data, name=f'MSE from {self.name}', sqrt=sqrt)


class MSESeries:
    """
    Wrapper class for pandas Series. Stores Mean Squared Error for a SampleDict
    """
    def __init__(self, name: str = None, sqrt: bool = False, *args, **kwargs):
        if 'data' in kwargs.keys() and isinstance(kwargs['data'], pd.Series):
            self._df = kwargs['data']
        else:
            self._df = pd.Series(name=name, *args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name
        self.sqrt = sqrt

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        if isinstance(self._df[item], pd.Series):
            return type(self)(data=self._df[item], name=self.name, sqrt=self.sqrt)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        temp_df = self._df.to_frame(name=self.name)
        temp_df['units'] = [get_param_units(param)[1:-1] for param in self._df.index]
        if not self.sqrt:
            temp_df['units'] = [temp_df['units'][param] if temp_df['units'][param] == r'ø'
                                else temp_df['units'][param] + r'^2'
                                for param in self._df.index]
        return temp_df.to_markdown()


class MSEDataFrame:
    """
    Wrapper class for pandas DataFrame. Stores Mean Squared Error for a SampleSet
    """
    def __init__(self, name: str = None, sqrt: bool = False, *args, **kwargs):
        if 'data' in kwargs.keys() and isinstance(kwargs['data'], pd.DataFrame):
            self._df = kwargs['data']
        else:
            self._df = pd.DataFrame(*args, **kwargs)
        if name is None:
            name = type(self).__name__
        self.name = name
        self.sqrt = sqrt

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._df, attr)

    def __getitem__(self, item):
        if isinstance(self._df[item], pd.Series):
            return MSESeries(data=self._df[item], name=self.name, sqrt=self.sqrt)
        if isinstance(self._df[item], pd.DataFrame):
            return type(self)(data=self._df[item], name=self.name, sqrt=self.sqrt)
        return self._df[item]

    def __setitem__(self, item, data):
        self._df[item] = data

    def __repr__(self):
        return self._df.__repr__()
    # def __repr__(self):
    #     temp_df = self._df.to_frame()
    #     temp_df['units'] = [get_param_units(param)[1:-1] for param in self._df.index]
    #     if not self.sqrt:
    #         temp_df['units'] = [temp_df['units'][param] if temp_df['units'][param] == r'ø'
    #                             else temp_df['units'][param] + r'^2'
    #                             for param in self._df.index]
    #     return temp_df.to_markdown()

    def mean(self, *args, **kwargs):
        # May resort to implement these one by one (at least the most useful)
        return MSESeries(data=self._df.mean(*args, **kwargs), name=self.name, sqrt=self.sqrt)


# Could be moved to a config object later
def set_hist_style(style: str = 'bilby') -> dict:
    available_styles = ['bilby', ]
    if style == 'bilby':
        return {
            # For plt.hist
            'bins': 50,
            'histtype': 'step',
            'density': True,
            'color': 'tab:blue',

            # For various axvlines
            'truth_color': 'orange',
            'truth_format': '-',
            'average_color': 'C0',
            'average_format': '-',
            'errorbar_color': 'C0',
            'errorbar_format': '--'

        }
    elif style == 'deserted':
        return {
            # For plt.hist
            'bins': 50,
            'histtype': 'step',
            'density': True,
            'color': 'khaki',

            # For various axvlines
            'truth_color': 'orange',
            'truth_format': '-',
            'average_color': 'gold',
            'average_format': '-',
            'errorbar_color': 'gold',
            'errorbar_format': '--'

        }
    else:
        raise ValueError(f'style \'{style}\' not understood. This are the available styles: {available_styles}')