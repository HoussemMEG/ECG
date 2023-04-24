import matplotlib.pyplot as plt
import numpy as np

from plot import Plotter
from reader import Reader
from utils import read_parameters

session = ["2023-04-06 66;66 200 of each small window",
           "2023-04-06 66;66 100 of each",
           "2023-04-06 66;66 only healthy and RBBB",
           "2023-04-06 66;66",
           "2023-04-06 66;66 visualisation"][-1]

condition = ['1dAVb', 'RBBB', 'LBBB', 'HEALTHY', 'SB', 'AF', 'ST', ]
# condition = ['HEALTHY']

batch_size = 49
reader = Reader(batch_size=batch_size * len(condition))
plotter = Plotter()
parameters = read_parameters(session)

data = []
all_category = []

for x_train, y_train in reader.read_hdf5(session, 'train', condition, random=True):
    data.append([np.mean(x_train[y_train == category], axis=0) for category in [0, 1, 2, 6, 3, 4, 5]])
    del x_train, y_train
    # data = np.array([np.mean(x_train[y_train == category], axis=0) for category in list(set(y_train))])


data = np.mean(np.array(data), axis=0)
plotter.plot_control_only(x=data, freq=parameters['model_freq'], condition=condition, show=True, save=False, vmax=0.02)
# plotter.plot_control_only(x=data, freq=parameters['model_freq'], condition=condition, show=True, save=False, vmax=0.02)

plt.show()
