import random
import string

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Slider
from scipy.stats import gaussian_kde
import seaborn as sns

import utils
from utils import index_to_time, truncate_colormap, set_ticks

# Use QT5agg backend for matplotlib
matplotlib.use('QT5agg')

# Font update for all figures
# font = {#'family': 'Latin Modern Roman',
#         'weight': 'normal',
#         'size': 12}
# matplotlib.rc('font', **font)

# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 12
#
# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# Ticks size modification
# matplotlib.rcParams['xtick.major.size'] = 7
# matplotlib.rcParams['xtick.major.width'] = 1.3
# matplotlib.rcParams['xtick.minor.size'] = 4
# matplotlib.rcParams['xtick.minor.width'] = 1.1
# matplotlib.rcParams['ytick.major.size'] = 7
# matplotlib.rcParams['ytick.major.width'] = 1.3
# matplotlib.rcParams['ytick.minor.size'] = 4
# matplotlib.rcParams['ytick.minor.width'] = 1.1
# matplotlib.rcParams.update({'font.size': 11})

# Slider container
sliders = []

class Plotter:
    LEADS = np.array(['DI', 'DII', 'DIII', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])

    def __init__(self, **kwargs):
        """
        Initializes a Plotter object with given arguments
        :param: kwargs: dictionary containing optional arguments
        """
        self._save = kwargs.get('save')  # save figure flag
        self._show = kwargs.get('show')  # show figure flag
        self._save_path = kwargs.get('save_path', './figures')  # path to save figures

    def plot_signal(self, x, meta_data=None, r_peaks=None, select_lead=None, fs=400, scaling=0.25, sep=100/12):
        """
        Plots the given ECG signals for the selected leads
        :param: x: shape: (N_subject, N_points, N_leads) input signal array
        :param: meta_data: shape: (N_subject, N_column) metadata for the input signals
        :param: r_peaks: shape: (N_subject, [N_peak]) R peaks position in an array form, not plotted if not provided
        :param: select_lead: list of leads to plot, or None to plot all leads
        :param: fs: sampling frequency of the signal
        :param: scaling: scaling factor for the signal
        :param: sep: separation distance between the leads
        """
        if not self._show and not self._save:
            return

        if x.shape[0] > 10:
            # raise error if the number of plots is too large
            raise RuntimeError('The number of plots is large <{:}>and should be lowered for'
                               ' computational purposes'.format(x.shape[0]))
        N = x.shape[1]
        t = np.linspace(0, (N-1)/fs, N)

        exam_id = self._init_exam_id(meta_data, len(x))
        # set mask to include only selected leads
        mask = np.in1d(self.LEADS, select_lead) if select_lead else np.ones_like(self.LEADS, dtype=bool)
        r_peaks = index_to_time(r_peaks, fs)

        for i, subj_ecg in enumerate(x[..., mask]):
            # create a new figure and reset the axis for each subject
            fig, ax = self._reset_ecg_fig(sep, mask, title=exam_id[i], time=t)
            for j, signal in enumerate(subj_ecg.T):
                # plot each lead signal separately
                ax.plot(t, signal/scaling + j * sep, color='blue', linewidth=1)

                # plot red vertical lines on each R-peak
                if r_peaks:
                    for peak in r_peaks[i]:
                        ax.vlines(peak, ymin=(j - 0.45) * sep, ymax=(j + 0.45) * sep,
                                  color='red', linewidth=1, linestyles='dashed', alpha=0.7)

            fig_name = f'{exam_id[i]}_raw_ECG' if not r_peaks else f'{exam_id[i]}_processed_ECG'
            self._save_figure(fig, fig_name)

    def plot_epochs(self, epochs, meta_data=None, select_lead=None, before=0.2, after=0.4, plot_mean=True):
        """
        This method plots the segmented ECG signal for each subject in the provided epochs list.
         The ECG signal is plotted in a grid format, where each subplot corresponds to a specific lead of the ECG.
         The method also optionally plots the mean ECG signal for each lead across all epochs for a given subject.

        :param: epochs: list of length [N_subject] that contain the segmented ECG of shape (N_segment, N_point, N_lead).
        :param: meta_data: list of metadata corresponding to each subject.
        :param: select_lead: list of ECG leads to plot. If None, all leads will be plotted.
        :param: before: time in seconds to plot before the R-peak.
        :param: after: time in seconds to plot after the R-peak.
        :param: plot_mean: boolean indicating whether to plot the mean ECG signal for each lead across all epochs.
        :return: /
        """
        if not self._show and not self._save:
            return

        if len(epochs) > 10:
            # raise error if the number of plots is too large
            raise RuntimeError('The number of plots is large <{:}>and should be lowered for'
                               ' computational purposes'.format(len(epochs)))
        N = epochs[0].shape[1]
        t = np.linspace(-before, after, N, endpoint=False)

        exam_id = self._init_exam_id(meta_data, len(epochs))
        # set mask to include only selected leads
        mask = np.in1d(self.LEADS, select_lead) if select_lead else np.ones_like(self.LEADS, dtype=bool)

        for i, subj_ecg in enumerate(epochs):  # shape (n_epoch, n_point, n_lead)
            # create a new figure and reset the axis for each subject
            fig, axes = self._reset_epochs_fig(n_plots=sum(mask), title='exam_id: ' + exam_id[i],
                                               xlabel='Time [s]', ylabel='Amplitude [V]')

            for j, lead_signals in enumerate(subj_ecg[..., mask].T):
                ax = fig.add_subplot(axes[j])
                ax.set_title(self.LEADS[mask][j])
                for signal in lead_signals.T:
                    ax.plot(t, signal, linewidth=0.7, color='blue', alpha=0.5)
                if plot_mean:
                    ax.plot(t, np.mean(lead_signals, axis=1), linewidth=1.2, color='red')
            self._save_figure(fig, exam_id[i] + '_epochs')

    @staticmethod
    def _init_exam_id(meta_data, length):
        """
        Returns the exam ID from the metadata, or generates a random ID if metadata is None
        :param: meta_data: metadata for the input signals
        :param: length: length of the signal array
        :return: exam ID(s)
        """
        if meta_data is not None:
            return meta_data['exam_id'].apply(str).to_numpy()
        return ['NE-' + ''.join(random.choice(string.ascii_uppercase) for _ in range(8)) for _ in range(length)]

    def _reset_ecg_fig(self, sep, mask, time, title=None):
        """
        Reset the layout of the plots for the signal of each subject
        :return: Figure and Axes
        """
        fig, ax = plt.subplots(1, 1, figsize=(19.2 / 1.3, 10.8 / 1.3), constrained_layout=True)
        n_lead = sum(mask)

        ticks = np.linspace(0, (n_lead-1) * sep, n_lead)
        labels = self.LEADS[mask]
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Time [s]', fontsize=11)
        ax.set_ylabel('Amplitude [V]')
        ax.set_title('exam_id: ' + title)

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.grid(True, which='minor', linestyle='dotted', linewidth=0.4, alpha=0.4)
        ax.xaxis.grid(True, which='major', linestyle='dotted', linewidth=1, alpha=0.4)
        ax.yaxis.grid(True, which='major', linestyle='dotted', linewidth=0.4, alpha=0.4)

        ax.set_ylim(-sep, n_lead * sep)
        ax.set_xlim(time[0] - 0.2, time[-1] + 0.2)
        return fig, ax

    @staticmethod
    def _reset_epochs_fig(n_plots, title, xlabel, ylabel):
        fig: plt.Figure
        ax: plt.Axes

        if n_plots > 1:
            fig = plt.figure(figsize=(19.2 / 1.3, 10.8 / 1.3), constrained_layout=True)
            gs0 = matplotlib.gridspec.GridSpec(2, 1, figure=fig)
            gs = [*gs0[0].subgridspec(1, n_plots // 2 + n_plots % 2), *gs0[1].subgridspec(1, n_plots // 2)]
        else:
            fig, gs = plt.subplots(constrained_layout=True)
            gs = [gs]

        fig.suptitle(title)
        fig.supxlabel(xlabel, fontsize=11)
        fig.supylabel(ylabel)
        return fig, gs

    @staticmethod
    def plot_alpha(alphas, coef_path, residue_path, fig_name, show=True, save=False, selection=None):
        """
        Plot the residue: epsilon = |y - \hat{y}|^2 (+ |\beta|_1 or _2) for a varying level or sparsity in blue.
        The number of features used by the model is displayed in red.
        :param: alphas: the sparsity levels (returned from LASSO-Lars).
        :param: coef_path: virtual modal stimuli for different sparsity levels.
        :param: residue_path: epsilon value for different sparsity levels.
        :param: fig_name: saving figure name.
        :param: show: True in order to show the plotted figures.
        :param: save: True in order to save the figures.
        :param: selection: Contain the selected levels of sparsity by the algorithm and mark them in the figure. If None,
                           no marks are added.
        :return: /
        """
        # Plot alpha decay in top sub-plot
        fig, axes = plt.subplots(figsize=(19.2 / 1.3, 10.8 / 1.3), nrows=2, layout='constrained')
        axes[0].plot(alphas)
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("$\\alpha$")
        axes[0].xaxis.set_minor_locator(AutoMinorLocator())
        axes[0].xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        axes[0].xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        axes[0].yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        if selection:
            axes[0].scatter(selection, alphas[selection], c='k', marker='x', zorder=9, alpha=0.7)

        # Plot epsilon in blue in bottom sub-plot
        color = 'tab:blue'
        axes[1].plot(residue_path, color=color, zorder=1)
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("$\\frac{1}{2N}|y-\\hat{y}|_2^2 + \\alpha|\hat{u}|$", fontsize=13, color=color)
        axes[1].tick_params(axis='y', labelcolor=color)
        axes[1].xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        axes[1].xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        if selection:
            axes[1].scatter(selection, residue_path[selection], c='k', marker='x', zorder=9, alpha=0.7)

        # Plot of the number of non-zero entries of \beta in red in bottom sub-plot
        color = 'tab:red'
        ax = axes[1].twinx()
        ax.plot(np.count_nonzero(coef_path, axis=0), color=color, zorder=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("# of parameters",color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        if selection:
            ax.scatter(selection, np.count_nonzero(coef_path, axis=0)[selection],
                       c='k', marker='x', zorder=9, alpha=0.7)

        _save_figure(fig, fig_name + '-residue.png', path='./figures/', save=save, show=show)

    @staticmethod
    def plot_y_hat(y, y_hat, ax=None, fig=None):
        """
        Plot the real signal along with the predicted signal y_hat.
        :param y: real EEG signal to be displayed.
        :param y_hat: predicted EEG signal (output of our model) to be displayed.
        :param ax: plot on this ax if provided, else create a new one.
        :param fig: plot on this fig if provided, else create a new one.
        :return: line of (y_hat) which is important for a smooth slider update.
        """
        if ax is None or fig is None:
            fig, ax = plt.subplots(figsize=[12, 8])

        ax.plot(y, label='Y_true')
        line, = ax.plot(y_hat, 'C3', label='Y_pred')

        ax.legend()
        ax.set_xlabel("Time [index]")
        ax.set_ylabel("Amplitude [$\mu$V]")
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.xaxis.grid(True, which='minor', linestyle=':', linewidth=0.3)
        ax.xaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        ax.yaxis.grid(True, which='major', linestyle='-', linewidth=0.4)
        return line

    @staticmethod
    def plot_control(x=None, y=None, y_hat=None, freq=None, cmap='blue_red', fig_name=None,
                     show=True, save=False, vmax=0.1):
        """
        :param: x: generated SDF along alpha path, shape (n_freq, n_path).
        :param: y: real signal shape (n_point,).
        :param: y_hat: predicted signal along alpha path, shape (n_point, n_path).
        :param: freq: frequency of the model.
        :param: cmap: color map can only be ['bleu_red' or 'black_white']. 'black_white' show an illuminating white
                      points where the VMS are non-zero, the 'blue_red' show a gradient from blue (negative value)
                      to red (positive value) of the VMS while white represent 0.
        :param: fig_name: saving figure name.
        :param: show: True in order to show the plotted figures.
        :param: save: True in order to save the figures.
        :param: vmax: The highest absolute value over which the VMS are considered at max or mix, so their color is blue
                      or red (instead of being a color in between).
        :return: /
        """
        # Some verifications
        if (y is None) ^ (y_hat is None):
            raise ValueError("The real signal y and the predicted signal y_hat should be given simultaneously "
                             "but got instead y = {:} and y_hat = {:}".format(y, y_hat))

        # Setting some parameters
        n_feature = x.shape[0]
        n_freq = len(freq)
        n_point = len(y) if y is not None else int(n_feature / n_freq)

        # Color map choice and settings
        cmap_choice = cmap
        cmap, vmin, vmax, x = utils.get_cmap(cmap, x, vmax)

        # reshaping the data
        try:
            x = np.swapaxes(np.array(np.split(x, n_point, axis=0)), 0, 1)
        except ValueError:
            n_point -= 1
            y = y[1:]
            y_hat = y_hat[1:]
            x = np.swapaxes(np.array(np.split(x, n_point, axis=0)), 0, 1)

        fig, axes = plt.subplots(nrows=2, figsize=(19.2 / 1.6, 10.8 / 1.6), dpi=100)
        fig.subplots_adjust(bottom=0.15, left=0.075, right=0.96, top=0.95)

        # 1) plotting y and y_hat
        line = Plotter.plot_y_hat(y, y_hat[:, -1], ax=axes[0], fig=fig)

        # 2) plotting the activations
        mat = axes[1].imshow(x[..., -1], interpolation='nearest', aspect='auto', origin='upper',  # 'auto'  'equal'
                             cmap=cmap, vmax=vmax, vmin=vmin, zorder=1)
        axes[1].xaxis.set_minor_locator(AutoMinorLocator())
        y_ticks = set_ticks(n_freq)
        axes[1].set_yticks(y_ticks)
        y_labels = ['{:1.1f}'.format(freq[int(idx)]) if 0 <= int(idx) < n_freq else 'pb' for idx in axes[1].get_yticks()]
        axes[1].set_yticklabels(y_labels)
        axes[1].set_xlabel('Time [index]')
        axes[1].set_ylabel('Model Frequency [Hz]')
        axes[1].tick_params(top=True, direction='out', which='both')
        axes[1].yaxis.set_minor_locator(AutoMinorLocator())
        axes[1].tick_params(which='major', length=6)

        # add slider that changes the level of fit depending on alpha values
        def update_wave(val):
            idx = int(slider.val)
            line.set_ydata(y_hat[:, idx])
            mat.set_data(x[..., idx])
            fig.canvas.draw_idle()
            return mat

        slider = Slider(plt.axes([0.25, 0.05, 0.5, 0.03]), 'Alpha value.', 0, y_hat.shape[-1]-1, valinit=y_hat.shape[-1]-1, valfmt='%d')
        slider.on_changed(update_wave)
        _save_figure(fig, fig_name, path='./figures/', save=save, show=show)
        sliders.append(slider)


    @staticmethod
    def plot_control_only(x=None, freq=None, cmap='blue_red', condition=None, fig_name=None, filter=False,
                          select=(None, None, None, None), show=True, save=False, vmax=0.1):
        """
        :param: x: generated SDF along alpha path, shape (n_freq, n_path).
        :param: y: real signal shape (n_point,).
        :param: y_hat: predicted signal along alpha path, shape (n_point, n_path).
        :param: freq: frequency of the model.
        :param: cmap: color map can only be ['bleu_red' or 'black_white']. 'black_white' show an illuminating white
                      points where the VMS are non-zero, the 'blue_red' show a gradient from blue (negative value)
                      to red (positive value) of the VMS while white represent 0.
        :param: fig_name: saving figure name.
        :param: show: True in order to show the plotted figures.
        :param: save: True in order to save the figures.
        :param: vmax: The highest absolute value over which the VMS are considered at max or mix, so their color is blue
                      or red (instead of being a color in between).
        :return: /
        """
        # Setting some parameters
        n_category = x.shape[0]
        n_lead = x.shape[1]
        n_feature = x.shape[2]
        n_path = x.shape[-1]
        n_freq = len(freq)
        n_point = int(n_feature / n_freq)

        # Color map choice and settings
        cmap, vmin, vmax, x = utils.get_cmap(cmap, x, vmax)

        # reshaping the data to (n_category, n_target, n_freq, n_point, n_path)
        try:
            x = np.transpose(np.split(x, n_point, axis=2), (1, 2, 3, 0, 4))
        except ValueError:
            x = np.transpose(np.split(x, n_point - 1, axis=2), (1, 2, 3, 0, 4))

        # Selection box delimiters
        xmin = select[0] if select[0] is not None else 0
        xmax = select[1] if select[1] is not None else x.shape[3]
        ymin = select[2] if select[2] is not None else 0
        ymax = select[3] if select[3] is not None else x.shape[2]


        fig, axes = plt.subplots(nrows=n_category, figsize=(19.2 / 1.6, 10.8 / 1.6), dpi=100)
        fig.subplots_adjust(bottom=0.1, left=0.075, right=0.90, top=0.98)

        # 1) plotting the activations
        mats = []
        if n_category == 1:
            axes = [axes]

        for i in range(n_category):
            if filter:
                for j in range(n_lead):
                    x[i, j, :, :, -1] = utils.convolution2d(x[i, j, :, :, -1], kernel_type='gaussian')

            if any(select):
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linestyle='--', fill='forestgreen',
                                     lw=1, color='forestgreen', alpha=0.2)
                axes[i].add_patch(rect)

            mat = axes[i].imshow(x[i, 0, ..., -1], interpolation='nearest', aspect='auto', origin='upper',  # 'auto'  'equal'
                                 cmap=cmap, vmax=vmax, vmin=vmin)
            mats.append(mat)
            # from index to temporal x-axis
            axes[i].set_xticks(np.arange(0, 240, 40))
            axes[i].set_xticklabels(['0.2', '-0.1', '0.0', '0.1', '0.2', '0.3'])

            axes[i].xaxis.set_minor_locator(AutoMinorLocator())
            # axes[i].xaxis.set_minor_formatter('{x:.0f}')
            y_ticks = set_ticks(n_freq)
            axes[i].set_yticks(y_ticks)
            y_labels = ['{:1.1f}'.format(freq[int(idx)]) if 0 <= int(idx) < n_freq else 'pb' for idx in axes[i].get_yticks()]
            axes[i].set_yticklabels(y_labels)
            axes[i].set_ylabel(f'{condition[i]} \n Frequency [Hz]')
            axes[i].tick_params(top=True, direction='out', which='both')
            axes[i].yaxis.set_minor_locator(AutoMinorLocator())
            axes[i].tick_params(which='major', length=5)
            if i == n_category - 1:
                axes[i].set_xlabel('Time [index]')
            else:
                axes[i].set_xticklabels([])

        # add slider that changes the level of fit depending on alpha values
        def update_wave(val):
            idx_lead = int(slider_lead.val)
            idx_path = int(slider_path.val)
            for i in range(n_category):
                mats[i].set_data(x[i, idx_lead, ..., idx_path])
            fig.canvas.draw_idle()
            return mat

        slider_lead = Slider(plt.axes([0.95, 0.2, 0.02, 0.7]), 'lead.', 0, n_lead - 1, valinit=0, valfmt='%d', orientation='vertical')
        slider_lead.on_changed(update_wave)
        slider_path = Slider(plt.axes([0.25, 0.03, 0.5, 0.02]), 'Alpha value.', 0, n_path - 1, valinit=n_path - 1, valfmt='%d')
        slider_path.on_changed(update_wave)

        _save_figure(fig, fig_name, path='./figures/', save=save, show=show)
        sliders.append(slider_path)

    def plot_psd(self, Pxx, freq, title=None):
        """
        to be completed
        """
        Pxx_mean, Pxx_std = np.mean(Pxx, axis=0), np.std(Pxx, axis=0)
        ci = 1.96 * Pxx_std / np.sqrt(len(Pxx))  # confidence interval

        if Pxx_mean.ndim == 1:
            Pxx_mean = Pxx_mean[np.newaxis, :]
            ci = ci[np.newaxis, :]

        fig, axes = self._reset_epochs_fig(n_plots=Pxx_mean.shape[0], title='PSD', xlabel='Frequencies [Hz]',
                                           ylabel='$V^2/Hz$')

        leads = self.LEADS if title is None else [title]
        for i in range(len(Pxx_mean)):
            ax = fig.add_subplot(axes[i])
            ax.set_title(leads[i])
            ax.plot(freq, Pxx_mean[i], color='b')
            ax.fill_between(freq, (Pxx_mean[i] - ci[i]), (Pxx_mean[i] + ci[i]), color='b', alpha=.1)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.xaxis.grid(True, which='minor', linestyle='dotted', linewidth=0.4, alpha=0.4)
            ax.xaxis.grid(True, which='major', linestyle='dotted', linewidth=1, alpha=0.4)
            ax.yaxis.grid(True, which='major', linestyle='dotted', linewidth=0.4, alpha=0.4)

        self._save_figure(fig, figure_name="PSD_all_leads_N_" + str(len(Pxx)))

    def plot_psd_frequency_fetching(self, cdf, freq, x, y, y_find, output_freq):
        """
        to be completed
        """
        fig, ax = plt.subplots(1, 1, figsize=(19.2 / 2, 10.8 / 2))
        ax.plot(x, y, 'r')
        ax.plot(freq, cdf, '.', color='b')
        ax.set_xlim(0, freq[-1])
        ax.set_ylim(0, 1)
        ax.xaxis.set_minor_locator(AutoMinorLocator())

        # Plot horizontal lines
        for i, val in enumerate(y_find):
            plt.plot([0, output_freq[i]], [val, val], '-', color='k', linewidth=0.5, alpha=0.15)
        # Plot vertical lines
        for i, val in enumerate(y_find):
            plt.plot([output_freq[i], output_freq[i]], [0, val], '-', color='k', linewidth=0.5, alpha=0.15)

        self._save_figure(fig, figure_name='PSD frequencies projection')

    @staticmethod
    def correlogram(x, y, show=False, save=False):
        """
        Create a correlogram of the features x
        :param x: (np.array) features, shape (n_samples, n_features)
        :param y: (np.array) labels, shape (n_samples, )
        :param: show: True in order to show the plotted figures.
        :param: save: True in order to save the figures.
        :return: /
        """
        if not show and not save:
            return

        # Get the number of features and labels
        n_features = x.shape[1]
        labels = list(set(y))

        # Get a color map and set colors for each label
        cmap = matplotlib.cm.get_cmap('Set1')
        colors = [cmap(x) for x in np.arange(0, len(labels))]
        # colors = [cmap(x) for x in np.linspace(0, 1, len(labels), endpoint=True)]  # if not quantitative cmap

        fig, axes = plt.subplots(n_features, n_features, figsize=(19.2 / 1.5, 10.8 / 1.5), tight_layout=True)

        # If there is only one feature, plot a scatter plot
        if n_features == 1:
            for species, color in zip(labels, colors):
                data = x[y == species]
                axes.scatter(data, np.zeros_like(data), alpha=0.9, color=color)
            return

        # For multiple features
        for i in range(n_features):
            for j in range(n_features):
                # If this is the lower-triangle, add a scatter plot for each group
                for species, color in zip(labels, colors):
                    data = x[y == species]
                    outliers = utils.is_outlier(data[:, j]) + utils.is_outlier(data[:, i])
                    if i > j:  # Scatter plot on lower triangle
                        axes[i, j].scatter(data[~outliers, j], data[~outliers, i], linewidths=0, color=color, alpha=0.6, s=20)
                    elif i == j:  # Add KDE on the main diagonal
                        p = sns.kdeplot(data[~outliers, j], ax=axes[i, j], alpha=0.7, color=color, bw_method='scott')
                        p.set(ylabel=None)
                    else:
                        # Blank axes
                        axes[i, j].axis('off')

                # Set labels for x-axis and y-axis
                if j == 0:
                    axes[i, j].set_ylabel('Feature ' + str(i + 1))
                if i == n_features - 1:
                    axes[i, j].set_xlabel('Feature ' + str(j + 1))

                # Remove ticks for both axis
                axes[i, j].set_xticks([])
                axes[i, j].set_yticks([])

        # Add title and save figure
        fig.suptitle('Correlogram', fontsize=16)
        _save_figure(fig, figure_name='correlogram', save=save, show=show)

    def _save_figure(self, fig, figure_name):
        # utils.enlarge_axis_limits(fig)  # enlarge axis limits
        if self._save:
            path = f'{self._save_path}/{figure_name}.pdf'  # .svg'
            fig.savefig(path, bbox_inches='tight')
            print("Figure saved at : {:}".format(path))
        if not self._show:
            plt.close()

def _save_figure(fig, figure_name, path='./figures', save=False, show=False):
    # utils.enlarge_axis_limits(fig)  # enlarge axis limits
    if save:
        path = f'{path}/{figure_name}.pdf'  # .svg'
        fig.savefig(path, bbox_inches='tight')
        print("Figure saved at : {:}".format(path))
    if not show:
        plt.close()

