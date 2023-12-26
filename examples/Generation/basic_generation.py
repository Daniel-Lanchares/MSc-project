# -*- coding: utf-8 -*-
from pathlib import Path
from dtempest.gw.generation.parallel import Injector


if __name__ == '__main__':

    seed = 15
    zero_pad = 3

    files_path = Path('/home/daniel/Documentos/GitHub/MSc-files')
    inj = Injector(2000, files_path / f'Raw Datasets/Raw_Dataset_{seed:0{zero_pad}}.pt', seed=seed,
                   config={'num_workers': 4, 'chunksize': 5})

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
    Seed 006, 5 workers, chunksize 05: Acceptance 1042/2000 (52.10%) It took 05 minutes 24 seconds, 5.24 it/s
    Seed 007, 4 workers, chunksize 10: Acceptance 1053/2000 (52.65%) It took 05 minutes 04 seconds, 6.64 it/s
    Seed 008, 4 workers, chunksize 15: Acceptance 1100/2000 (55.00%) It took 05 minutes 05 seconds, 6.63 it/s
    Seed 009, 4 workers, chunksize 05: Acceptance 1146/2000 (57.30%) It took 05 minutes 33 seconds, 6.07 it/s
    Seed 010, 4 workers, chunksize 05: Acceptance 1083/2000 (54.15%) It took 05 minutes 11 seconds, 6.51 it/s
    Seed 011, 4 workers, chunksize 05: Acceptance 1058/2000 (52.90%) It took 05 minutes 04 seconds, 6.65 it/s
    Seed 012, 4 workers, chunksize 05: Acceptance 1051/2000 (52.55%) It took 05 minutes 01 seconds, 6.73 it/s
    Seed 013, 4 workers, chunksize 05: Acceptance 1039/2000 (51.95%) It took 05 minutes 19 seconds, 6.35 it/s
    Seed 014, 4 workers, chunksize 05: Acceptance 1067/2000 (53.35%) It took 05 minutes 20 seconds, 6.32 it/s
    Seed 015, 4 workers, chunksize 05: Acceptance 1072/2000 (53.60%) It took 05 minutes 22 seconds, 6.28 it/s
    '''
