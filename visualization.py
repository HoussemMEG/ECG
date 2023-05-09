import time

import matplotlib.pyplot as plt
import numpy as np

from plot import Plotter
from reader import Reader
from utils import read_parameters, convolution2d

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
           "2023-04-06 66;66 test 10 small windows",
           "2800 per category _ 20 freq _ short window",
           "2800 per category _ 35 freq _ long window"][-1]

tic = time.perf_counter()

condition = ['1dAVb', 'HEALTHY', 'SB', 'ST']
all_category = [0, 6, 3, 5]

# condition = ['1dAVb', 'RBBB', 'LBBB', 'HEALTHY', 'SB', 'AF', 'ST']
# all_category = [0, 1, 2, 6, 3, 4, 5]


batch_size = 40
plotter = Plotter()
parameters = read_parameters(session)

diff = True
compare_healthy = True
filter_ = False and diff
focus_category = 6
focus_category = all_category.index(focus_category)
vmax = 0.02 if not diff else 0.005


if compare_healthy:
    counter = 0
    reader_ = Reader(batch_size= 100 * len(condition))
    hdf5_batch_iterator = reader_.read_hdf5(session, 'validation', ['HEALTHY'], random=True)
    n_exams, n_leads, n_features, n_pars = next(hdf5_batch_iterator)
    data_healthy = np.zeros((1, 12, n_features))
    for x_healthy, _ in hdf5_batch_iterator:
        data_healthy += np.mean(x_healthy[..., -1], axis=0)
        counter += 1

    data_healthy /= counter



counter = 0
reader = Reader(batch_size=batch_size * len(condition))
hdf5_batch_iterator = reader.read_hdf5(session, 'train', condition, random=True)
n_exams, n_leads, n_features, n_pars = next(hdf5_batch_iterator)

data = np.zeros((len(all_category), n_leads, n_features, n_pars))

for x_train, y_train in hdf5_batch_iterator:
    # ## test conv2d before mean
    # temp = np.transpose(np.array(np.split(x_train, parameters['n_point'] - 1, axis=2)), (1, 2, 0, 3, 4))
    # for i in range(len(x_train)):
    #     for j in range(n_leads):
    #         # reshaping the data
    #         temp[i, j, -5, -10, -1] = 0.01
    #         temp[i, j, :, :, -1] = convolution2d(temp[i, j, :, :, -1], kernel_type='gaussian')
    # x_train  = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2] * temp.shape[3], temp.shape[4]))
    # ##end of testing

    for i, category in enumerate(all_category):
        data[i] += np.mean(x_train[y_train == category], axis=0)

    counter += 1
    if counter == 10:
        break
del x_train, y_train

data /= counter
# data /= reader.n_split

print('reading time {:.1f} s = {:.1f} min'.format(time.perf_counter() - tic, (time.perf_counter() - tic) / 60))

if diff and not compare_healthy:
    temp = data[focus_category].copy()
    for i, category in enumerate(all_category):
        data[i] -= temp

## testing new idea
if diff and compare_healthy:
    for i, category in enumerate(all_category):
        for j in range(len(parameters['parsimony'])):
            data[i, :, :, j] -= data_healthy[0, :, :]


plotter.plot_control_only(x=data, freq=parameters['model_freq'], cmap='blue_red', condition=condition,
                          filter=filter_, select=(50, 78, 0, 16),  # (time_min, time_max, freq_min, freq_max)
                          show=True, save=False, vmax=vmax)

plt.show()

