import torch
from pathlib import Path

# Look for more colors just in case
class PrintStyle:

    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'


def identity(x):
    return x


def check_format(dataset):
    """
    Allows functions to be given either a dataset tensor or its path.

    Parameters
    ----------
    dataset : str, pathlib.Path or torch.tensor.

    Returns
    -------
    dataset : torch.tensor.

    """
    if isinstance(dataset, (str, Path)):
        dataset = torch.load(dataset)
    return dataset
