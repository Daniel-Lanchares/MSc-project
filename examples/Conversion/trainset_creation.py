from pathlib import Path

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_1',
    'chi_2',
    'luminosity_distance',
    'theta_jn',
    'ra',
    'dec',
    'phase',
    'psi'
]

for start in range(0, 60, 10):
    seeds = list(range(start, start+10))
    dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
    dataset.change_parameter_name('d_L', to='luminosity_distance')

    # Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
    dataset = convert_dataset(dataset, params_list)
    dataset.save(trainset_dir / '10_sets' / (f'{seeds[0]} to {seeds[-1]}.'+', '.join(params_list)+'.pt'))
    del dataset
