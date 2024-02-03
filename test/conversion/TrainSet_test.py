from pathlib import Path
from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset
from dtempest.core.conversion_utils import TrainSet

dataset_path = Path('/media/daniel/easystore/Daniel/MSc-files/Raw Datasets')
trainset_path = Path('/media/daniel/easystore/Daniel/MSc-files/Trainsets')
seeds = range(2)
dataset = load_rawsets(dataset_path, seeds2names(seeds))

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    'd_L',
    'ra',
    'dec'
]

# Convert to TrainSet object (pd.DataFrame based instead of OrderedDict based)
# Reusing variable wil optimize RAM if RawSet not necessary
trainset = convert_dataset(dataset, params_list)
trainset.save(trainset_path/'example_l.pt', True)

loaded = TrainSet.load(trainset_path/'example_l.pt')
print(trainset.name)
print(loaded.name)