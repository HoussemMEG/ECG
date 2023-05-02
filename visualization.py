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
           "2023-04-06 66;66 visualisation 2000 35 freq",
           "2023-04-06 66;66 visualisation 500 35 freq short window",
           "2023-04-06 66;66 test 25 short window",
           "2023-04-06 66;66 test 10 small windows"][-5]

condition = ['1dAVb', 'HEALTHY', 'SB', 'ST']
all_category = [0, 6, 3, 5]

# condition = ['1dAVb', 'RBBB', 'LBBB', 'HEALTHY', 'SB', 'AF', 'ST']
# all_category = [0, 1, 2, 6, 3, 4, 5]


batch_size = 1
reader = Reader(batch_size=batch_size * len(condition))
plotter = Plotter()
parameters = read_parameters(session)

diff = True
filter_ = True and diff
focus_category = 6
focus_category = all_category.index(focus_category)
vmax = 0.02 if not diff else 0.02
data = np.zeros((len(all_category), 12, parameters['n_features'], len(parameters['parsimony'])))  # (7, 12, 8365, 50)


## new idea use other healthy data and do the difference
counter = 0
data_healthy = np.zeros((1, 12, parameters['n_features'], 1))  # (7, 12, 8365, 50)
for x_healthy, _ in reader.read_hdf5(session, 'validation', ['HEALTHY'], random=False):
    data_healthy += np.mean(x_healthy[:, :, :, [-1]], axis=0)
    del x_healthy
    counter += 1
data_healthy /= counter


counter = 0
tic = time.perf_counter()
for x_train, y_train in reader.read_hdf5(session, 'train', condition, random=True):
    for i, category in enumerate(all_category):
        data[i] += np.mean(x_train[y_train == category], axis=0)
    del x_train, y_train

    counter += 1
    if counter == 1:
        break

# data /= reader.n_split
data /= counter

print('reading time {:.1f} s = {:.1f} min'.format(time.perf_counter() - tic, (time.perf_counter() - tic) / 60))

# if diff:
#     temp = data[focus_category].copy()
#     for i, category in enumerate(all_category):
#         data[i] -= temp

## new idea difference
if diff:
    for i, category in enumerate(all_category):
        for pars in range(data.shape[-1]):
            data[i, :, :, pars] -= data_healthy[0, :, :, -1]

plotter.plot_control_only(x=data, freq=parameters['model_freq'], cmap='blue_red', condition=condition,
                          filter=filter_, select=(55, 70, 9, 35),  # (time_min, time_max, freq_min, freq_max)
                          show=True, save=False, vmax=vmax)

plt.show()

