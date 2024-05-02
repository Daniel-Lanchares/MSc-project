from pathlib import Path

from dtempest.core.common_utils import load_rawsets, seeds2names
from dtempest.gw.conversion import convert_dataset

files_dir = Path('/media/daniel/easystore/Daniel/MSc-files')
rawdat_dir = files_dir / 'Raw Datasets'
trainset_dir = files_dir / 'Trainsets'

params_list = [
    'chirp_mass',
    'mass_ratio',
    'chi_eff',
    # 'chi_1',
    # 'chi_2',
    'luminosity_distance',
    'theta_jn',
    'ra',
    'dec',
    # 'phase',
    # 'psi'
]
step = 20
for start in range(100, 120, step):
    seeds = list(range(start, start+step))
    dataset = load_rawsets(rawdat_dir, seeds2names(seeds))
    dataset.change_parameter_name('d_L', to='luminosity_distance')

    # Convert to Trainset object (pd.DataFrame based instead of OrderedDict based)
    dataset = convert_dataset(dataset, params_list)
    dataset.save(trainset_dir / f'{step}_sets' / (f'{seeds[0]} to {seeds[-1]}.'+', '.join(params_list)+'.pt'))
    del dataset
