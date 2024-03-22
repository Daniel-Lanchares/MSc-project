import numpy as np
import pandas as pd

from dtempest.core.train_utils import TrainSet

'''
This is a mockup of the __getattr__ of the TrainSet class, useful to tinker around and try to integrate as many methods
as possible.
'''
method_debug = False
trainset_debug = True

if method_debug:
    df_methods = []
    s_methods = []
    fmt_methods = []
    misc = []
    weirds = []
    not_in = []
    for attr in pd.DataFrame.__dict__:
        try:
            if hasattr(getattr(pd.DataFrame, attr), '__annotations__'):
                returns = getattr(pd.DataFrame, attr).__annotations__['return'].split(' | ')
                if 'DataFrame' in returns:
                    df_methods.append(attr)
                    # return type(self)(data=getattr(self._df, attr), name=self.name)
                elif 'Series' in returns:
                    s_methods.append(attr)
                elif 'str' in returns:
                    fmt_methods.append(attr)
                else:
                    # print(f'{attr} returns {returns}')
                    misc.append(attr)
            else:
                # print(f'This one is weird {attr}')
                weirds.append(attr)
        except KeyError:
            # print(f'        {attr} not on __dict__')
            not_in.append(attr)
            pass

    for lis in [df_methods, s_methods, fmt_methods, misc, weirds, not_in]:
        lis.sort()
        print(f'{len(lis)} methods of {len(pd.DataFrame.__dict__)}')
        print(lis)

'''
This is the testbed for testing the TrainSet integration of pandas functionality, 
to be extended later to other wrappers.
'''

if trainset_debug:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(7, 5))
    train = TrainSet(data=pd.DataFrame(data=data,
                                       columns=['hey', 'have', 'a', 'nice', 'day']),
                     name='test')

    print('Pre-transform')
    print(train)
    train['new_column'] = abs(train.loc[train['hey'] < -0.5, 'hey'] * 2)
    print(train)
    # print(train.iloc[:] < 0)
    col = train.pop('a')
    print(f'\nI stole this along the way')
    print(col)
    applied = train[train.iloc[:] > 0.3].dropna(how='all').T

    print(f'\nname: {applied.name}, type: {type(applied)}\n')
    print('Post-transform')
    print(applied)
    print('\nOr in markdown if preferred\n')
    print(applied.to_markdown())
