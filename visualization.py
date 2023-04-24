import time

import matplotlib.pyplot as plt
import numpy as np

from plot import Plotter
from reader import Reader
from utils import read_parameters

session = ["2023-04-06 66;66 200 of each small window",
           "2023-04-06 66;66 100 of each",
           "2023-04-06 66;66 only healthy and RBBB",
           "2023-04-06 66;66",
           "2023-04-06 66;66 visualisation",
           "2023-04-06 66;66 visualisation 1000 10 freq",
           "2023-04-06 66;66 visualisation 100 70 freq",
           "2023-04-06 66;66 visualisation 2000 35 freq"][-2]

# condition = ['1dAVb', 'HEALTHY', 'SB', 'ST']
# all_category = [0, 6, 3, 5]

condition = ['1dAVb', 'RBBB', 'LBBB', 'HEALTHY', 'SB', 'AF', 'ST']
all_category = [0, 1, 2, 6, 3, 4, 5]


batch_size = 15
reader = Reader(batch_size=batch_size * len(condition))
plotter = Plotter()
parameters = read_parameters(session)

diff = True
diff_focus = 3
vmax = 0.02 if not diff else 0.001
data = np.zeros((len(all_category), 12, parameters['n_features'], len(parameters['parsimony'])))  # (7, 12, 8365, 50)

tic = time.perf_counter()
for x_train, y_train in reader.read_hdf5(session, 'train', condition, random=True):
    for i, category in enumerate(all_category):
        data[i] += np.mean(x_train[y_train == category], axis=0)
    del x_train, y_train
    break

data /= reader.n_split
print('reading time {:.1f} s = {:.1f} min'.format(time.perf_counter() - tic, (time.perf_counter() - tic) / 60))

if diff:
    temp = data[diff_focus].copy()
    for i, category in enumerate(all_category):
        data[i] -= temp

plotter.plot_control_only(x=data, freq=parameters['model_freq'], cmap='blue_red', condition=condition,
                          show=True, save=False, vmax=vmax)

plt.show()

"""
    # data = np.array([np.mean(x_train[y_train == category], axis=0) for category in list(set(y_train))])
"""
