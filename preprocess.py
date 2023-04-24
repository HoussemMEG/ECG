from typing import Tuple, List

import biosppy.signals.ecg as bs_ecg
import biosppy.signals.tools as st
import numpy as np
import pandas as pd
from biosppy.signals.tools import filter_signal

from utils import _filter_signal_mod, execution_time


class Preprocess:
    def __init__(self, fs=400, before=0.2, after=0.4, filt=(3, 45)):
        """
        Initialize the Preprocess class.

        :param: fs: Sampling frequency of the ECG signal (default=400).
        :param: before: Time in seconds to include before the R peak (default=0.2).
        :param: after: Time in seconds to include after the R peak (default=0.4).
        :param: filt: Frequency range for bandpass filtering. (default=(3, 45))
        """
        self.fs = fs
        st._filter_signal = _filter_signal_mod
        self.before = int(before * self.fs)
        self.after = int(after * self.fs)
        self._length = None
        self._filter = list(filt)
        self.parameters = self._store_parameters()

    @execution_time
    def filter(self, signal):
        """
        Filters a signal using a bandpass filter
        :param: signal (array): The signal to filter
        :return filtered: The filtered signal
        """
        order = int(0.3 * self.fs)
        filtered, _, _ = filter_signal(
            signal=signal,
            ftype="FIR",
            band="bandpass",
            order=order,
            frequency=self._filter,
            sampling_rate=self.fs)
        return filtered

    def extract_r_peaks(self, x, meta_data):
        """
        Extracts R peaks from ECG signals using the Hamilton QRS detector from <Lead I>,
            as it has been shown to work best for this lead.

        :param: x shape (N_subject, N_points, N_leads): input signal array.
        :return: result: list of ndarray: A list of 1D arrays containing the R peak indices for each input signal.
        """
        result, to_remove = [], []
        for i, subj_ecg in enumerate(x):
            # R peaks fetching then correction
            (r_peaks,) = bs_ecg.hamilton_segmenter(subj_ecg[:, 0], sampling_rate=self.fs)  # <Lead I> only
            (r_peaks,) = bs_ecg.correct_rpeaks(signal=subj_ecg[:, 0], rpeaks=r_peaks, sampling_rate=self.fs, tol=0.05)

            if not r_peaks.any():  # if no peaks found remove the data instance
                to_remove.append(i)
            else:
                result.append(r_peaks)

        x, meta_data = self._remove(x, meta_data, to_remove)
        return result, x, meta_data

    def _extract_heart_beats(self, x, meta_data, r_peaks):
        """
        Extracts ECG epochs centered around the R peaks for each subject in the input x.
            if over one lead the signal is zero the data instance is removed.
        :param: x: array-like, shape (N_subjects, N_points, N_leads) Input ECG signals for each subject.
        :param: meta_data: Metadata data frame associated with the input data.
        :param: r_peaks: list of ndarray, shape (N_subjects,) List of 1D arrays containing the R peak indices for each
                            subject's ECG signal.
        :return: epochs: list of length [N_subject] that contain the segmented ECG of shape (N_segment, N_point, N_lead)
                         for each subject.
                 new_r_peaks: a 1D list of [N_subject] that contain the new location of R-peak indices for each subject
                              after truncation of epochs to the available signal length.
        """
        # If the signal length is not defined, set it to the length of the input signal.
        if self._length is None:
            self._length = x.shape[1]

        epochs = []
        new_r_peaks = []
        to_remove = []

        # Extract epochs for each subject.
        for i, subj_ecg in enumerate(x):
            # Extract epochs for each R peak in the subject's ECG signal if the signal is within the limits.
            epoch = np.array([subj_ecg[r - self.before:r + self.after, :] for r in r_peaks[i]
                              if r - self.before >= 0 and r + self.after <= self._length])

            # If any epoch has all zeros, remove the corresponding subject's ECG signal.
            if any(epoch.sum(axis=0).sum(axis=0) == 0):
                to_remove.append(i)
            else:
                new_r_peaks.append(np.array([r for r in r_peaks[i]
                                             if r - self.before >= 0 and r + self.after <= self._length]))
                epochs.append(epoch)

        # Remove subjects with all zero epochs from the input data.
        self._remove(x, meta_data, to_remove)

        return x, new_r_peaks, epochs, meta_data

    @staticmethod
    def _remove(x, meta_data, to_remove):
        """
        Removes subjects with all zero epochs or no r_peaks detection from the input data and metadata.

        :param: x: Input ECG signals for each subject. Shape (N_subjects, N_points, N_leads).
        :param: meta_data: Metadata data frame associated with the input data.
        :param: to_remove: List of indices of subjects to be removed.
        :return
            tuple:
                x: Input ECG signals after removing subjects with all zero epochs.
                        Shape (N_remaining_subjects, N_points, N_leads).
                meta_data: Metadata associated with the input data after removing subjects with bad epochs.
        """
        x = np.delete(x, to_remove, axis=0)
        meta_data.drop(index=to_remove, inplace=True)
        meta_data.reset_index(drop=True, inplace=True)
        return x, meta_data

    def process(self, x, meta_data) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], pd.DataFrame]:
        """
        Preprocesses the input ECG signals by applying filtering, R peak extraction, and ECG epoch extraction.

        :param: x: Input ECG signals for each subject. Shape (N_subjects, N_points, N_leads).
        :param: meta_data: Metadata data frame associated with the input data.

        :returns:
            tuple:
                x: Preprocessed ECG signals for each subject. Shape (N_subjects, N_points, N_leads).
                r_peaks: List of 1D arrays containing the R peak indices for each subject's preprocessed ECG signal.
                epochs: List of segmented ECG epochs of shape (N_segment, N_point, N_lead) for each subject.
        """
        # Filtering
        x = self.filter(x)

        # R peak extraction from Lead I
        r_peaks, x, meta_data = self.extract_r_peaks(x, meta_data)

        # Extract epochs
        x, new_r_peaks, epochs, meta_data = self._extract_heart_beats(x, meta_data, r_peaks=r_peaks)

        return x, r_peaks, epochs, meta_data

    def _store_parameters(self):
        """
        Parameter to store
        """
        return {'fs': self.fs,
                'idx_before': self.before,
                'idx_after': self.after,
                'filter_low': self._filter}
