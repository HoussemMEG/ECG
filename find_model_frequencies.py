import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from plot import Plotter
from preprocess import Preprocess
from reader import Reader
from utils import compute_PSD

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
matplotlib.use('QT5agg')


# parameters
m_pendulum = 35
leads = np.array(['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

# init the class instances
reader = Reader(batch_size=500, n=1)
plotter = Plotter(show=True, save=True)
preprocess = Preprocess(fs=400, before=0.2, after=0.4)

condition = ['HEALTHY']  # ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']

tic = time.perf_counter()
for x, meta_data in reader.read(condition=condition, random=False):
    exams_id = meta_data['exam_id'].to_numpy()
    # plotter.plot_signal(x, meta_data=meta_data, select_lead=[])
    x, r_peaks, epochs, meta_data = preprocess.process(x, meta_data)
    # plotter.plot_signal(x, meta_data=meta_data, r_peaks=r_peaks, select_lead=[])
    # plotter.plot_epochs(epochs, meta_data=meta_data, select_lead=[])

    # Compute PSD / Spectrum and plot each lead separately
    Pxx, freq = compute_PSD(epochs, fs=400, cutoff=50)  # shape (n_epochs, n_channels, n_freqs)
    plotter.plot_psd(Pxx, freq)

    # Mean over all leads
    Pxx_leads = np.mean(Pxx, axis=0)
    plotter.plot_psd(Pxx_leads, freq, title='all_leads')

    # Cumulative distribution function
    pdf = np.mean(Pxx_leads, axis=0)
    cdf = np.cumsum(pdf)

    func = interpolate.interp1d(freq, cdf, kind='cubic', fill_value='extrapolate')
    x = np.linspace(0, freq[-1], 1000, endpoint=True)
    y = func(x)

    # Frequency projection and finding
    y_find = np.linspace(freq[0], 1, m_pendulum + 1, endpoint=True)
    y_find = y_find[1:-1]

    output_freq = []
    for to_find in y_find:
        f = x[np.argmin(np.abs(y - to_find))]
        output_freq.append(f)
    temp = output_freq[-1]
    val = (freq[-1] - output_freq[-1]) / 3
    output_freq.append(temp + val)
    output_freq.append(temp + 2.2 * val)
    y_find = np.append(y_find, [func(output_freq[-2]), func(output_freq[-1])])

    output_freq = [round(n, 3) for n in output_freq]
    print(output_freq[1:])
    print(len(output_freq[1:]))
    # Plot
    plotter.plot_psd_frequency_fetching(cdf, freq, x, y, y_find, output_freq)

    break

plt.show()

print('Running time {:.2f}'.format(time.perf_counter() - tic), flush=True)

"""
            # (f, S) = scipy.signal.periodogram(y[:, 0], fs=400, scaling='density')
            # plt.plot(f, S, label='scipy')
            #
            # (f, S) = scipy.signal.welch(y[:, 0], fs=400, nperseg=8*1024)
            # plt.plot(f, S, label='welch')
"""
