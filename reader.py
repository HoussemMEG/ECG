import h5py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tqdm import tqdm

from utils import decompress_hdf5


class Reader:
    _SETS = {'test', 'validation', 'train', 'learning'}
    TEST_FILES = ['exams_part4.hdf5', 'exams_part8.hdf5', 'exams_part12.hdf5']
    VALIDATION_FILES = ['exams_part2.hdf5', 'exams_part9.hdf5', 'exams_part14.hdf5']
    ALL_CONDITIONS = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']

    def __init__(self, data_path=None, batch_size=500, stratified=False, set_name=None, **kwargs):
        """
        Main reading class that returns the data of the healthy subjects along with the specified target condition.
            The data are returned along with their meta_data: {'exam_id', 'age', 'is_male', 'condition', 'normal_ecg',
            'trace_file'}
        :param: batch_size (int): The batch size that we want to be returned for each iteration.
        :param: stratified:
        :param: set_name:
        :param: data_path (str): The path of the folder that contains the data if not specified we look directly in
                                    './data' for the data and 'exams.csv' is supposed to be present in that folder.
        """
        self._path = data_path if data_path is not None else './data'
        self._batch_size = batch_size
        self._n = kwargs.get('n', 1)
        self._stratified = stratified
        self.conditions = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']  # do not modify

        # Read and prepare the description dictionary that contains all the subjects information
        self.description = pd.read_csv(self._path + '/exams.csv')
        self.description.drop(columns=['nn_predicted_age', 'death', 'timey'], inplace=True)
        self._remove_multiple_disease()
        self._select_set(set_name)
        self.description.sort_values(by='trace_file', inplace=True, ignore_index=True)

    def _select_set(self, set_name):
        """
        Selects a specific set of data from the description dataframe based on the set name.

        :param: set_name: (str) The name of the set to select. Possible values are 'test', 'validation', 'train', and 'learning'.
        :return: None
        """
        if set_name is None:  # if set_name is None, all data is read
            return

        # create a boolean mask that filters out rows where 'trace_file' is or is not in sets, pay attention to "~"
        if set_name == 'test':
            filter_ = ~self.description['trace_file'].isin(self.TEST_FILES)
        elif set_name == 'validation':
            filter_ = ~self.description['trace_file'].isin(self.VALIDATION_FILES)
        elif set_name == 'train':
            filter_ = self.description['trace_file'].isin(self.TEST_FILES + self.VALIDATION_FILES)
        elif set_name == 'learning':
            filter_ = self.description['trace_file'].isin(self.TEST_FILES)
        else:
            raise ValueError('Allowed sets {:} but {:} were given'.format(self._SETS, set_name))

        self.description.drop(self.description[filter_].index, inplace=True)

    def _remove_multiple_disease(self):
        """
        Remove the exams that have multiple labels (diseases)
        :return: /
        """
        filter_ = self.description[self.conditions].apply(sum, axis=1) > 1
        self.description.drop(self.description[filter_].index, inplace=True)

    def read(self, condition=None, random=False) -> (np.ndarray, pd.DataFrame):
        """
        Main reading method, it returns the data of the healthy subjects and the subjects that are affected by
            'condition' type.
        :param: condition:  The condition(s) of the subjects that we want to target. The healthy subjects are
                    always returned.
        :param: random (bool): True to select randomly one batch of experiments, False for normal behaviour.
        :return: x (ndarray): shape (batch_size, 4096, 12) contain the ECG signal of 'batch_size' subjects.
        :return: meta_data: shape (batch_size) is the meta_data that goes along with 'x' and contain:
                {'exam_id', 'age', 'is_male', 'condition', 'normal_ecg', 'trace_file'}
        """
        # Create a new column that contains if the subject is healthy or no
        is_healthy = ~self.description[self.conditions].any(axis=1).to_numpy()
        self.description.insert(9, 'HEALTHY', is_healthy)
        self.conditions.append('HEALTHY')

        # Transform condition columns into one <label> column and delete old condition columns
        label_columns = self.description.columns[3:-3]
        self.description.insert(3, 'label', self.description[label_columns].dot(label_columns))
        self.description.drop(columns=label_columns, inplace=True)

        # Keep only the target condition on the description file
        if condition:
            filter_ = self.description['label'].isin(condition)
            self.description = self.description[filter_].reset_index(drop=True)

        # Select randomly <n x batch_size> exams if random = True
        #   if stratified = True, select <n x batch_size> for each class
        if random:
            if self._stratified:
                self.description = self.description.groupby('label', group_keys=False).\
                    apply(lambda z: z.sample(n=self._n * self._batch_size))
                self._batch_size *= len(self.description['label'].unique())
            else:
                self.description = self.description.sample(n=self._n * self._batch_size)

            self.description = self.description.sort_values(by=['trace_file'], ignore_index=True)

        # Init all the exams to be read
        exam_to_read, file_to_read = self.description['exam_id'].to_numpy(), self.description['trace_file'].to_numpy()
        n_exams = len(exam_to_read)

        # Batch reading
        add_batches = 1 if n_exams % self._batch_size == 0 else 2
        for i in range(1, (n_exams // self._batch_size) + add_batches):
            start = (i - 1) * self._batch_size
            end = min(i * self._batch_size, n_exams)
            exam_id = exam_to_read[start:end]
            files = file_to_read[start:end]

            x = np.zeros((len(exam_id), 4096, 12))
            meta_data = pd.DataFrame()

            idx_switch = [0] + [k for k in range(1, len(files)) if files[k-1] != files[k]] + [len(files)]
            for j in range(1, len(idx_switch)):  # If in the same batch we need to read multiple files
                with h5py.File(self._path + '/' + files[idx_switch[j-1]], 'r') as data:
                    filter = np.array([True if elem in exam_id[idx_switch[j-1]:idx_switch[j]] else False for elem in data['exam_id']])
                    x[idx_switch[j-1]:idx_switch[j]] = data['tracings'][filter]
                    meta_data = pd.concat((meta_data, self.description.set_index('exam_id').loc[data['exam_id'][filter]]))

                    if filter.sum() != idx_switch[j]-idx_switch[j-1]:
                        print('\033[91m{:} Subject not found and is replaced by zeros'.format(
                            filter.sum() - (idx_switch[j]-idx_switch[j-1])), '\033[0m')
            yield x, meta_data.reset_index()

    def read_test(self, condition=None, index=0):
        """
        Method to read test data and parse the labels
        :param: condition: list of conditions to include in the annotations file
        :param: index: integer index to select the annotations file to read
        :return: test_data: shape (N_subject, N_point=4096, N_lead) ECG data.
                 annotations: shape (N_subject, len(condition)) annotations that goes along with the ECG data that
                 specifies each subjects condition after selection.
        """
        # Read the right annotations from the right csv file depending on the index
        read_from = ['cardiologist1.csv', 'cardiologist2.csv', 'cardiology_residents.csv', 'dnn.csv',
                     'emergency_residents.csv', 'gold_standard.csv', 'medical_students.csv'][index]

        annotation = pd.read_csv(self._path + '/test/annotations/' + read_from)

        # Keep only the healthy subjects and the target condition on the annotation file
        if condition:
            all_cond = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
            for cond in condition:
                all_cond.remove(cond)
            filter_ = ~(annotation[all_cond]).any(axis=1)
            annotation = annotation[filter_].drop(columns=all_cond)

        # Read all the test data of ecg_tracings.hdf5 test data and return only the healthy + target condition exams
        with h5py.File(self._path + '/test/ecg_tracings.hdf5', 'r') as data:
            if 'filter' in locals():
                return data['tracings'][filter_], annotation  # return filtered data along with the annotations
            return data['tracings'][:], annotation  # return all data along with the annotations

    def read_hdf5(self, session, set_name, condition=None, random=False):
        """
        Read features and labels from an HDF5 file in a stratified manner.

        :param: session: The name of the session directory where the HDF5 file is located
        :param: random: Whether to randomize the order of the exam data (default False)
        :return: A tuple of two numpy arrays, containing features and labels
        """
        file_path = f'./features/{session}/{set_name}_features.hdf5'
        with h5py.File(file_path, 'r') as f:
            # Get the list of all conditions in the file
            all_conditions = [x.decode('utf-8') for x in f['conditions']]

            # Select the data that corresponds to the appropriate condition (label)
            if condition:
                temp = [(key, val[0]) for key, val in f['labels'].items() if all_conditions[val[0]] in condition]
                exams, labels = zip(*temp)
            else:
                exams, labels = list(f['features'].keys()), [val[0] for val in f['labels'].items()]

            # Get dimension parameters from the data
            n_exams = len(exams)
            n_targets = len(f['features'][exams[0]]['u']['idx'].keys())
            n_features = f['features'][exams[0]]['u']['n_features'][0]
            n_path = f['features'][exams[0]]['u']['n_path'][0]

            # Generate splits for the exams based on the batch size
            add_batch = 0 if n_exams % self._batch_size == 0 else 1
            n_split = len(exams) // self._batch_size + add_batch
            if n_split > 1:
                skf = StratifiedKFold(n_splits=n_split, shuffle=random)
                splits = skf.split(exams, labels)
            else:
                splits = [range(len(exams))]

            # Batch reading
            for k, exams_to_read in enumerate(splits):
                if len(exams_to_read) == 2:
                    exams_to_read = exams_to_read[-1]
                x = np.zeros((len(exams_to_read), n_targets, n_features, n_path))
                y = np.empty((len(exams_to_read),), dtype=int)
                for i, exam_idx in enumerate(tqdm(exams_to_read, desc=f'Batch {k+1}/{n_split}')):
                    features, x0 = decompress_hdf5(f['features'][exams[exam_idx]])
                    x[i] = features
                    y[i] = labels[exam_idx]

                yield x, y
