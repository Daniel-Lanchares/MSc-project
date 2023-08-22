# -*- coding: utf-8 -*-

from CBC_estimator.generation import Injector

seed = 0
inj = Injector(2000, f'Dataset/Raw_Dataset_{seed}.pt', seed=seed)

'''
Seed 0, 4 workers: Acceptance 1072/2000 (53.60%) It took 29 minutes 1 seconds
Seed 0, 4 workers: Acceptance 1072/2000 (53.60%) It took 29 minutes 36 seconds, 1.13 it/s (with progressbar)
'''
