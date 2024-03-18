# -*- coding: utf-8 -*-
from pathlib import Path
from dtempest.gw.generation.parallel import Injector

if __name__ == '__main__':

    seeds = range(77, 81)
    zero_pad = 3

    files_path = Path('/media/daniel/easystore/Daniel/MSc-files/Raw Datasets')
    for seed in seeds:
        inj = Injector(2000, seed=seed, config={'log_file': files_path / 'log.txt',
                                                'seed_zero_pad': zero_pad})
        inj.save(files_path / f'Raw_Dataset_{seed:0{zero_pad}}.pt')

    '''
    Seed 0, 4 workers, chunksize 01: Acceptance 1072/2000 (53.60%) It took 29 minutes 01 seconds
    Seed 0, 4 workers, chunksize 01: Acceptance 1072/2000 (53.60%) It took 29 minutes 36 seconds, 1.13 it/s(progressbar)
    
    New installation
    Seed 000, 4 workers, chunksize 05: Acceptance 1072/2000 (53.60%) It took 11 minutes 15 seconds, 2.98 it/s
    Seed 001, 4 workers, chunksize 01: Acceptance 1044/2000 (52.20%) It took 11 minutes 57 seconds, 2.80 it/s
    Seed 002, 4 workers, chunksize 08: Acceptance 1052/2000 (52.60%) It took 11 minutes 20 seconds, 2.96 it/s
    Seed 003, 2 workers, chunksize 05: Acceptance 1042/2000 (52.10%) It took 11 minutes 19 seconds, 2.96 it/s
    
    Better RAM setup
    Seed 004, 4 workers, chunksize 05: Acceptance 1081/2000 (54.05%) It took 05 minutes 07 seconds, 6.59 it/s
    Seed 005, 6 workers, chunksize 05: Acceptance 1028/2000 (51.40%) It took 06 minutes 30 seconds, 5.17 it/s
    
    ...
    
    Seed 999, 4 workers, chunksize 05: Acceptance 1064/2000 (52.50%) It took 05 minutes 21 seconds, 6.31 it/s
    '''
