import bisect
import time
from multiprocessing import Queue
from typing import List, Tuple

import matplotlib
import numpy as np
import scipy
import scipy.signal as ss
import termcolor
from matplotlib.mlab import psd
from scipy.integrate import simpson
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# old stuff imports
import os
import json
from scipy.stats import kurtosis, skew
from tabulate import tabulate  # newly installed


def execution_time(function):
    def my_function(*args, **kwargs):
        start_time = time.time()
        result = function(*args, **kwargs)
        if time.time() - start_time > 1e-2:
            text = "Execution time of " + function.__name__ + ": {0:.3f} (s)".format(time.time() - start_time)
        else:
            text = "Execution time of " + function.__name__ + ": {0:.3f} (ms)".format(1000*(time.time()-start_time))
        print_c(text, 'blue')
        return result

    return my_function


def print_c(text, color=None, highlight=None, bold=False):
    if bold:
        if color is None:
            print(termcolor.colored(text, attrs=['bold']))
        else:
            print(termcolor.colored(text, color, attrs=['bold']))
    else:
        if highlight:
            print(termcolor.colored(text + '\033[1m{:}\033[0m'.format(highlight), color))
        else:
            print(termcolor.colored(text, color))


def _filter_signal_mod(b, a, signal, zi=None, check_phase=True, **kwargs):
    """
    Modification of _filter_signal in biosppy signals such that it takes 3D input and filter the 'axis=1' dimension
    For more details check biosppy/signals/tools.py
    """
    # check inputs
    if check_phase and zi is not None:
        raise ValueError(
            "Incompatible arguments: initial filter state cannot be set when \
            check_phase is True."
        )

    axis = 1 if signal.ndim == 3 else 0

    if zi is None:
        zf = None
        if check_phase:
            filtered = ss.filtfilt(b, a, signal, axis, **kwargs)
        else:
            filtered = ss.lfilter(b, a, signal, axis, **kwargs)
    else:
        filtered, zf = ss.lfilter(b, a, signal, axis, zi=zi, **kwargs)

    return filtered, zf


def index_to_time(arr, fs):
    if arr is None:
        return None

    for i in range(len(arr)):
        arr[i] = [elem/fs for elem in arr[i]]
    return arr


def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
    """
    Create a new color map from a segment of an existing colormap defined by the minimum and maximum values.
    :param: cmap: The original colormap to truncate.
    :param: min_val: The minimum value of the segment to extract, between 0 and 1.
    :param: max_val: The maximum value of the segment to extract, between 0 and 1.
    :param: n: The number of colors in the new colormap.
    :return: A new truncated colormap.
    """
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min_val, b=max_val),
        cmap(np.linspace(min_val, max_val, n, endpoint=True)))
    return new_cmap


def set_ticks(n_freq, n_batch=None):
    """
    # create repeated ticks for control_plot method
    :param n_freq:
    :param n_batch:
    :return:
    """
    if n_batch is not None:  # case where I group features
        extra_ticks = [(i + 1) * n_freq - 1 for i in range(n_batch)]
        if n_freq < 6:
            ticks = list(np.arange(0, n_batch * n_freq, n_freq)) + extra_ticks
        elif n_freq < 20:
            ticks = list(np.arange(0, n_batch * n_freq, int(n_freq / 5))) + extra_ticks
        else:
            ticks = list(np.arange(0, n_batch * n_freq, int(n_freq / 5))) + extra_ticks  # / 10 here
        return list(set(ticks))
    else:  # case where I plot one by one
        if n_freq < 6:
            ticks = list(np.arange(0, n_freq, n_freq)) + [n_freq - 1]
        elif n_freq < 20:
            ticks = list(np.arange(0, n_freq, int(n_freq / 5))) + [n_freq - 1]
        else:
            ticks = list(np.arange(0, n_freq, int(n_freq / 5))) + [n_freq - 1]  # / 10 here
        return list(set(ticks))


# Transform safely ndarray to list for JSON saving
def ndarray_to_list(arr: List[np.ndarray]):
    for target_idx in range(len(arr)):
        if arr[target_idx] is not None:
            arr[target_idx] = arr[target_idx].tolist()
    return arr


# Compress the features to take less space (since they are really sparse)
def compress(features: List[np.ndarray]) -> dict:
    idx = []  # shape: (n_target, (n_non_zero, n_non_zero))
    val = []  # shape: (n_target, n_non_zero)
    n_path = []  # shape: (n_target)
    for target_idx in range(len(features)):
        idx_i, idx_j = np.nonzero(features[target_idx])
        idx.append([idx_i, idx_j])
        val.append(features[target_idx][np.nonzero(features[target_idx])])
        n_path.append(features[target_idx].shape[1])
    compressed_features = {'idx': idx, 'val': val, 'n_path': n_path, 'n_features': features[0].shape[0]}
    return compressed_features


# Decompress the features from the compressed format
def decompress(compressed_features: dict) -> List[np.ndarray]:
    decompressed_features = []
    n_target = len(compressed_features['idx'])
    n_features = compressed_features['n_features']
    for target_idx in range(n_target):
        idx = compressed_features['idx'][target_idx]  # shape: (n_target, (n_non_zero, n_non_zero))
        val = compressed_features['val'][target_idx]  # shape: (n_target, n_non_zero)
        n_path = compressed_features['n_path'][target_idx]  # shape: (n_target)
        features = np.zeros((n_features, n_path))  # shape: (n_features, n_path)
        features[tuple(idx)] = val
        decompressed_features.append(features)
    return decompressed_features


def decompress_hdf5(feature_group):
    u = feature_group['u']
    n_target = len(u['idx'].keys())
    n_features = u['n_features'][0]
    n_path = u['n_path'][0]

    decompressed_features = np.zeros((n_target, n_features, n_path))
    for target_idx in range(n_target):
        idx = u['idx'][str(target_idx)]
        decompressed_features[target_idx, idx[0], idx[1]] = u['val'][str(target_idx)]
    return decompressed_features, feature_group['x0']  # shape (n_target, n_features, n_path)


def duplicate_element(input_list: list, indexes: list, repeat: int) -> list:
    """
    A function that duplicates elements from a list at given indices.

    :param input_list: The original list.
    :param indexes: The list of indices where elements will be duplicated.
    :param repeat: The number of times to duplicate the entire list if necessary.
    :return: A new list with duplicated elements.
    """
    output_list = input_list.copy()  # Create a copy of the original array to avoid modifying it.

    if repeat > 0:
        indexes = indexes + list(np.repeat(input_list, repeats=repeat))

    for idx in sorted(indexes, reverse=True):
        output_list.insert(bisect.bisect_left(output_list, idx), idx)
    return output_list


def compute_PSD(epochs, fs=400, cutoff=50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Power Spectral Density (PSD) of a set of signals.

    :param: epochs: A 3D array of shape (n_epochs, n_leads, n_samples).
    :param: fs: The sampling frequency of the signals.
    :param: cutoff: The cutoff frequency used to truncate the PSD.

    :return:
        Pxx_container: PSD container with shape (n_epochs, n_leads, n_freqs),
            where n_freqs is the number of frequency bins up to the cutoff frequency.
    """
    Pxx_container = []
    # for each subject's ECG
    for epoch in epochs:
        Pxx_leads = []
        epoch_mean = np.mean(epoch, axis=0)
        for lead in epoch_mean.T:
            # Pxx, freq = psd(lead, Fs=fs)  # matplotlib PSD compute
            freq, Pxx = scipy.signal.periodogram(lead, fs=fs, scaling='density')  # 'density', 'spectrum'
            # freq, Pxx = scipy.signal.welch(lead, fs=fs, nperseg=8 * 1024)

            # Find index of cutoff frequency
            idx = np.argmin(np.abs(freq - cutoff))
            freq, Pxx = freq[:idx], Pxx[:idx]

            # Compute area in order to make the computed PSD as a density
            # area = simpson(Pxx, freq)
            area = np.sum(Pxx)
            Pxx = Pxx / area

            Pxx_leads.append(Pxx)
        Pxx_container.append(np.array(Pxx_leads))

    return np.array(Pxx_container), freq


def init_working_q(epochs, exams_id, cpu_count):
    """
    Initialize the working queue with the work to be done.

    :param: epochs: A list of 3D numpy arrays, where each array contains the epoched ECG with the
                shape (n_peaks, n_point, n_lead).
    :param: exams_id: A list of exam IDs corresponding to the epochs.
    :param: cpu_count: The number of CPU cores to use for processing the work.
    :return: A Queue object containing the work to be done.
    """
    working_q = Queue()
    # load work to be done in the working queue, each signal is sent with its corresponding exam_id
    for i, y in enumerate(epochs):
        y = np.mean(y, axis=0)
        working_q.put((exams_id[i], y))
    # add sentinel values to signal the end of the queue
    for _ in range(cpu_count):
        working_q.put(None)

    return working_q


def output_q_handler(output_q, meta_data, cpu_count, saver, conditions):
    """
    Process the results of the output queue after feature generation.

    :param: output_q: A Queue object containing the results of feature generation.
    :param: meta_data: A pandas DataFrame containing metadata about the input data.
    :param: cpu_count: The number of CPU cores used for processing.
    :param: saver: Saver instance to save the parsed and compressed output data.
    :param: conditions: A list of conditions.
    :return: feat_gen_param: feature generation parameters to be saved in the session (for results replication)
    """
    none_count = 0
    while True:
        res = output_q.get()
        if res is not None and none_count != cpu_count:  # if features encountered
            (exam_id, features, x0) = res
            label_name = meta_data.loc[meta_data.exam_id == exam_id, 'label'].to_numpy()
            label = conditions.index(label_name)
            saver.save(exam_id, features, x0, label)
        else:  # ending sentinel value
            none_count += 1
            if none_count == cpu_count:
                feat_gen_param = output_q.get()
                break

    return feat_gen_param


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# Smooth the resulting validation score and select upon the smoothed results (for a better stability of the algorithm)
def smoothing_scores(score, smoothing=True):
    if not smoothing:
        return score
    temp = [(score[i - 1] + 4 * score[i] + score[i + 1]) / 6 for i in range(1, len(score) - 1)]
    return np.array([(score[0] + score[1]) / 2, *temp, (score[-2] + score[-1]) / 2])


def read_parameters(session):
    """
    Read and set the right parameter from the param_files
    :param session: the working session
    :return: parameters
    """
    # Reading the JSON files
    parameters = {}
    # for file in os.listdir(f'./features/{session}'):
    for file in os.listdir(f'D:/ECG/features/{session}'):
        if file in ['DFG_parameters.json', 'preprocessing_parameters.json']:
            # parameters_path = f'./features/{session}/{file}'
            parameters_path = f'D:/ECG/features/{session}/{file}'
            with open(parameters_path) as f:
                parameters.update(json.load(f))

    # Parameters reading
    model_freq = np.array(parameters['model_freq'])
    n_freq = parameters['n_freq']
    n_point = parameters['n_point']
    n_features = parameters['n_features']
    pars = parameters['selection'] if parameters['selection'] is not None else parameters.get('selection_alpha', None)
    parameters['parsimony'] = pars
    data_case = parameters.get('data_case', 'evoked')
    alpha = parameters['alpha']
    version = parameters.get('version', 0)
    # Printing
    print_c('\nSessions: {:}'.format(session), 'blue', bold=True)
    print_c(' Data case: ', highlight=data_case)
    print_c(' Version: ', highlight=str(version))
    print_c(' Alpha: ', highlight=alpha)
    print(' Model frequencies: {:}'.format(model_freq))
    print_c(' N_freq = ', highlight=n_freq)
    print_c(' N_point = ', highlight=n_point)
    print_c(' Beta_dim = ', highlight=n_features)
    print(' Parsimony: {:}\n'.format(np.array(pars)))
    return parameters


def update_table(table, parsimony, scores, highlight_above):
    for pars_idx, pars in enumerate(parsimony):
        train_acc_pars = scores[pars_idx]
        if scores[pars_idx] >= highlight_above:
            table[0].append('\033[92m{:}\033[0m'.format(round(pars * 100)))
            table[1].append('\033[92m{:.1f}\033[0m'.format(train_acc_pars))
            table[-1].append('\033[92m OK\033[0m')
        else:
            table[0].append('{:}'.format(int(pars * 100)))
            table[1].append('{:.1f}'.format(train_acc_pars))
            table[-1].append('-')


def get_cmap(cmap, x, vmax):
    supported_cmap = ['black_white', 'blue_red', 'twilight']
    if cmap.lower() not in supported_cmap:
        raise ValueError('Plot control cmap_type supported {:} but {:} were given'
                         .format(supported_cmap, cmap.lower()))
    if cmap == 'black_white':
        cmap = ListedColormap(['k', 'w'], name='binary')
        x[x != 0] = 1
        vmin, vmax, = 0, 1

    elif cmap == 'twilight':
        plt.style.use('dark_background')
        cmap = plt.get_cmap('twilight').copy()
        cmap = truncate_colormap(cmap, 0.1, 0.9, 255)
        black_zero = list(map(cmap, range(255)))
        black_zero[127] = (0.0, 0.0, 0.0, 1.0)
        cmap = cmap.from_list('my_map', black_zero, N=255)
        vmin = -vmax

    elif cmap == 'blue_red':
        cmap = plt.get_cmap('seismic').copy()
        cmap = truncate_colormap(cmap, 0.2, 0.8, 255)
        white_zero = list(map(cmap, range(255)))
        white_zero[127] = (1.0, 1.0, 1.0, 1.0)
        cmap = cmap.from_list('my_map', white_zero, N=255)
        vmin = -vmax
    return cmap, vmin, vmax, x


def convolution2d(image, kernel_type, l=5, sigma=1):
    # Gaussian kernel
    if kernel_type == 'gaussian':
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)

    # Square kernel
    elif kernel_type == 'square':
        kernel = np.ones((l, l))

    kernel /= np.sum(kernel)
    m, n = kernel.shape

    y, x = image.shape
    output = np.zeros_like(image)
    image_padded = np.zeros((y + (m - 1), x + (n - 1)))
    image_padded[(m // 2):-(m // 2), (n // 2):-(n // 2)] = image

    for i in range(y):
        for j in range(x):
            output[i][j] = np.sum(image_padded[i:i+m, j:j+m] * kernel)

    return output

"""




OLD STUFF ADDED





"""


# mean of a list
def mean(lst):
    return sum(lst) / len(lst)


def prints(table, target, targets, k=None, feature=None, select_feature=None, timed=None):
    print_c('Target:   <{:}/{:}>'.format(target + 1, len(targets)), 'yellow', bold=True)
    # print_c('\t\tFeature: {:<20}     <{:}/{:}>\t\t{:.1f}s/it'.format(" / ".join(list(feature)), k + 1, len(select_feature), timed), 'blue', bold=True)
    print(tabulate(table, headers='firstrow', tablefmt="rounded_outline"))


def feature_extraction(x):
    """
    Feature extraction from the vector x
    :param x: the vector of VMS that we should extract the features from
    :return: dictionary, key are the features name, values are the extracted feature
    """
    dic = {'energy': np.sum(x ** 2, axis=1),
           'count_non_zero': np.count_nonzero(x, axis=1),
           'mean': np.mean(x, axis=1),
           'max': np.max(x, axis=1),
           'min': np.min(x, axis=1),
           'pk-pk': np.max(x, axis=1) - np.min(x, axis=1),
           'argmin': np.argmin(x, axis=1),
           'argmax': np.argmax(x, axis=1),
           'argmax-argmin': np.argmax(x, axis=1) - np.argmin(x, axis=1),
           'sum abs': np.sum(np.abs(x), axis=1),
           'var': np.var(x, axis=1),
           'std': np.std(x, axis=1),
           'kurtosis': kurtosis(x, axis=1),
           'skew': skew(x, axis=1),
           'count above mean': np.array([np.count_nonzero(row[np.where(row >= np.mean(np.abs(row)))]) for row in x]),
           'count below mean': np.array([np.count_nonzero(row[np.where(row <= np.mean(np.abs(row)))]) for row in x]),
           'max abs': np.max(np.abs(x), axis=1),
           'argmax abs': np.argmax(np.abs(x), axis=1),
           }

    # dimension checking
    for key, value in dic.items():
        if value.ndim > 1:
            raise ValueError("Feature {:} not extracted properly, has the dimension {:}".format(key, value.shape))
        if value.shape != (x.shape[0],):
            raise ValueError("Feature not corresponding to the right dimensions")
    return dic

def update_experience_values(experience_values, outer_memory, channel, stim):
    selection_validation = mean(outer_memory['selected validation'])
    if selection_validation > min(experience_values['validation_acc'][:]):
        i_min = np.argmin(experience_values['validation_acc'])
        experience_values['validation_acc'][i_min] = selection_validation
        experience_values['learning_category'][i_min] = np.array(outer_memory['learning_category'])
        experience_values['predicted_category'][i_min] = outer_memory['test_category']
        if selection_validation >= max(experience_values['validation_acc']):
            experience_values['channel'] = channel
            experience_values['stim'] = stim
            experience_values['test'] = mean(outer_memory['test'])
