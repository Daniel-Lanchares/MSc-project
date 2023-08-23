# -*- coding: utf-8 -*-
from pathlib import Path
from CBC_estimator.generation import Injector


if __name__ == '__main__':

    seed = 3
    files_path = Path('/home/daniel/Documentos/GitHub/MSc-files')
    inj = Injector(2000, files_path / f'Raw Datasets/Raw_Dataset_{seed}.pt', seed=seed,
                   config={'num_workers': 2})

    '''
    Seed 0, 4 workers, chunksize 1: Acceptance 1072/2000 (53.60%) It took 29 minutes 1 seconds
    Seed 0, 4 workers, chunksize 1: Acceptance 1072/2000 (53.60%) It took 29 minutes 36 seconds, 1.13 it/s (progressbar)
    
    New installation
    Seed 0, 4 workers, chunksize 5: Acceptance 1072/2000 (53.60%) It took 11 minutes 15 seconds, 2.98 it/s
    Seed 1, 4 workers, chunksize 1: Acceptance 1044/2000 (52.20%) It took 11 minutes 57 seconds, 2.80 it/s
    Seed 2, 4 workers, chunksize 8: Acceptance 1052/2000 (52.60%) It took 11 minutes 20 seconds, 2.96 it/s
    Seed 3, 2 workers, chunksize 5: Acceptance 1042/2000 (52.10%) It took 11 minutes 19 seconds, 2.96 it/s
    '''
