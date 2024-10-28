'''
Noise gathering:

There are much more sophisticated ways, I am sure.
It may be a poor connection on my end, but this takes a while
'''

from pathlib import Path
from gwpy.timeseries import TimeSeries



def query_noise(t, ifos, path, **fetch_kwargs):

        for ifo in ifos:
            try:
                strain = TimeSeries.fetch_open_data(ifo, t, t+500, **fetch_kwargs)
                strain.write(target=path/f'noise_{t}_{ifo}', format='hdf5')
            except ValueError:
                print(f'GWOSC has no data for time {t} on every detector requested (missing at least {ifo})')
                continue

if __name__ == '__main__':
    times = [1126257941, 1267928742]  # or the preferred times
    ifolist = ('L1', 'H1', 'V1')

    files_dir = Path('')  # Main directory
    noise_dir = files_dir / 'Noise'  # Noise directory

    [query_noise(t, ifolist, noise_dir, verbose=True) for t in times]