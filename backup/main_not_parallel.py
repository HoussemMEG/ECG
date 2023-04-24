import time

from multiprocessing import Queue, Process, current_process
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from featuregen import DFG
from plot import Plotter
from preprocess import Preprocess
from reader import Reader
import utils

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
matplotlib.use('QT5agg')


# parameters
condition = ['HEALTHY']  # ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']

# init the class instances
reader = Reader(batch_size=100, n=1)
plotter = Plotter(show=True, save=False)
preprocess = Preprocess(fs=400, before=0.2, after=0.4)
feature_gen = DFG(method='LARS',
                  f_sampling=400,
                  version=1,
                  alpha=1e-4,
                  find_alpha=False,
                  normalize=True,
                  model_freq=np.linspace(3, 45, 40, endpoint=True),
                  damping=None,  # (under-damped 0.008 / over-damped 0.09)
                  fit_path=True, ols_fit=True,
                  fast=True,
                  selection=0.01,
                  selection_alpha=None,
                  omit=None,  # omit either 'x0' or 'u' from y_hat computation
                  plot=(False, False), show=True, fig_name="fig name", save_fig=False,
                  verbose=True)

# main
if __name__ == '__main__':
    tic = time.perf_counter()
    for x, meta_data in reader.read(condition=condition, random=False):
        exams_id = meta_data['exam_id'].to_numpy()
        # plotter.plot_signal(x, meta_data=meta_data, select_lead=[])
        x, r_peaks, epochs = preprocess.process(x, meta_data)
        # plotter.plot_signal(x, meta_data=meta_data, r_peaks=r_peaks, select_lead=[])
        # plotter.plot_epochs(epochs, meta_data=meta_data, select_lead=[])

        # prepare containers and queues
        processes = []
        results = []

        # load work to be done in the working queue, each signal is sent with an exam_id
        for i, y in enumerate(epochs):
            y = np.mean(y, axis=0)
            features, x0 = feature_gen.generate(y)
            features, x0 = utils.compress(features), utils.ndarray_to_list(x0)
        break
    # print(results)
    print('Running time {:.2f}'.format(time.perf_counter() - tic), flush=True)




