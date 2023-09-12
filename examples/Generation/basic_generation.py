# -*- coding: utf-8 -*-
from pathlib import Path
from dtempest.gw.generation.parallel import Injector


if __name__ == '__main__':

    seed = 9
    files_path = Path('/home/daniel/Documentos/GitHub/MSc-files')
    inj = Injector(2000, files_path / f'Raw Datasets/Raw_Dataset_{seed}.pt', seed=seed,
                   config={'num_workers': 4, 'chunksize': 5})

    '''
    Seed 0, 4 workers, chunksize 01: Acceptance 1072/2000 (53.60%) It took 29 minutes 01 seconds
    Seed 0, 4 workers, chunksize 01: Acceptance 1072/2000 (53.60%) It took 29 minutes 36 seconds, 1.13 it/s(progressbar)
    
    New installation
    Seed 0, 4 workers, chunksize 05: Acceptance 1072/2000 (53.60%) It took 11 minutes 15 seconds, 2.98 it/s
    Seed 1, 4 workers, chunksize 01: Acceptance 1044/2000 (52.20%) It took 11 minutes 57 seconds, 2.80 it/s
    Seed 2, 4 workers, chunksize 08: Acceptance 1052/2000 (52.60%) It took 11 minutes 20 seconds, 2.96 it/s
    Seed 3, 2 workers, chunksize 05: Acceptance 1042/2000 (52.10%) It took 11 minutes 19 seconds, 2.96 it/s
    
    Better RAM setup
    Seed 4, 4 workers, chunksize 05: Acceptance 1081/2000 (54.05%) It took 05 minutes 07 seconds, 6.59 it/s
    Seed 5, 6 workers, chunksize 05: Acceptance 1028/2000 (51.40%) It took 06 minutes 30 seconds, 5.17 it/s
    Seed 6, 5 workers, chunksize 05: Acceptance 1042/2000 (52.10%) It took 05 minutes 24 seconds, 5.24 it/s
    Seed 7, 4 workers, chunksize 10: Acceptance 1053/2000 (52.65%) It took 05 minutes 04 seconds, 6.64 it/s
    Seed 8, 4 workers, chunksize 15: Acceptance 1100/2000 (55.00%) It took 05 minutes 05 seconds, 6.63 it/s
    '''
