import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot import Plotter
from preprocess import Preprocess
from reader import Reader

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
matplotlib.use('QT5agg')

# Init the class instances
reader = Reader(batch_size=200, n=10)
plotter = Plotter(show=True, save=False)
preprocess = Preprocess(fs=400, before=0.2, after=0.4)


# conditions to be read: ['HEALTHY', '1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
condition = ['ST']
cond_idx = ['HEALTHY', '1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST'].index(condition[0])

# leads : {DI, DII, DIII, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6}
for file in range(10):
    if 'ecg_data_set_' + str(file) + '.hdf5' not in os.listdir('./sample_data/'):
        with h5py.File('ecg_data_set_' + str(file) + '.hdf5', 'w') as f:
            f.create_dataset('x', shape=(0, 240, 12), maxshape=(None, 240, 12))
            f.create_dataset('y', shape=(0, 1), maxshape=(None, 1))

# main
file = 0
for x, meta_data in reader.read(condition=condition, random=True):
    # plotter.plot_signal(x, meta_data=meta_data, select_lead=[])
    x, r_peaks, epochs = preprocess.process(x)
    # plotter.plot_signal(x, meta_data=meta_data, r_peaks=r_peaks, select_lead=[])
    # plotter.plot_epochs(epochs, meta_data=meta_data, select_lead=[])
    container = []
    for j, subj_ecg in enumerate(epochs):
        if r_peaks[j].any():
            container.append(np.mean(subj_ecg, axis=0))
        else:
            continue
    container = np.array(container)

    with h5py.File('./sample_data/ecg_data_set_' + str(file) + '.hdf5', 'a') as f:
        f['x'].resize(f['x'].shape[0] + container.shape[0], axis=0)
        f['y'].resize(f['y'].shape[0] + container.shape[0], axis=0)
        f['x'][-container.shape[0]:] = container
        f['y'][-container.shape[0]:] = cond_idx * np.ones((container.shape[0], 1))
        # print(np.array(f['x']))
        print(f.keys(), f['x'], type(f['x'][0]), f['y'], '\n\n')
    file += 1

plt.show()


"""
    # Test data reading
    test_data, annotation = reader.read_test(condition=condition, index=5)  # index = 5 is the gold standard
    x, r_peaks, epochs = preprocess.process(test_data)
    plotter.plot_signal(x[0:1, ...], r_peaks=r_peaks[0:1], select_lead=[])
    plotter.plot_epochs(epochs[0:1], select_lead=[])
"""

"""
import h5py
import numpy as np

file = 0
with h5py.File('./ecg_data/ecg_data_set_' + str(file) + '.hdf5', 'r') as f:
    x = np.array(f['x'])
    y = np.array(f['y'])
    print(x.shape)
    print(y.shape)
"""