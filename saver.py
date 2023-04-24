import h5py
import os
import json

from utils import print_c


class Saver:
    def __init__(self, session, set_name, conditions, save=True, verbose=True):
        # parameters
        self._session = session
        self._verbose = verbose
        self._set_name = self._init_set_name(set_name)
        self._save = save

        # self._init_conditions(conditions)
        self._conditions = conditions
        self._init_hdf5()

    def save(self, exam_id, features, x0, label):
        """
        Saves generated features to the HDF5 file.

        :param: exam_id: (str) representing the ID of the exam.
        :param: features: dictionary containing the exams features.
        :param: x0: A numpy array containing the x0 component of the generated features.
        :param: label: a list containing the exams labels.
        :return: /
        """
        if not self._save:
            return

        with h5py.File(f'./features/{self._session}/{self._set_name}_features.hdf5', 'a') as f:
            # Create label dataset
            f['labels'].create_dataset(name=str(exam_id), data=[label])

            # Create features dataset
            global_group: h5py.Group = f['features'].create_group(name=str(exam_id))  # group that contains everything
            global_group.create_dataset(name='x0', data=x0)
            group = global_group.create_group(name='u')
            group_idx = group.create_group(name='idx')
            group_val = group.create_group(name='val')
            for target_idx in range(len(features['idx'])):
                group_idx.create_dataset(name=str(target_idx), data=features['idx'][target_idx])
                group_val.create_dataset(name=str(target_idx), data=features['val'][target_idx])
            group.create_dataset(name='n_path', data=features['n_path'])
            group.create_dataset(name='n_features', data=[features['n_features']])

    def save_json(self, data, file_name, verbose=True):
        """
        Save a dictionary as a JSON file.

        :param: data: The dictionary to save
        :param: file_name: The name of the JSON file
        :param: verbose: Whether to print a message indicating where the file was saved (default True)
        :return: /
        """
        if not self._save:
            return

        # Remove 'self' from dictionary if present
        data.pop('self', None)

        # Create directory for file if it doesn't exist
        directory = f'./features/{self._session}'
        os.makedirs(directory, exist_ok=True)

        # Save file
        file_path = os.path.join(directory, file_name + '.json')
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                existing_data = json.load(file)
                if not data.items() <= existing_data.items():
                    existing_data.update(data)
                    data = existing_data

        with open(file_path, 'w') as fp:
            json.dump(data, fp, indent=4)

            if verbose:
                print_c('JSON file <{: <27} saved at: {:}>'.format(file_name, directory), 'yellow', bold=True)

    def _create_folder(self):
        directory = f'./features/{self._session}'
        os.makedirs(directory, exist_ok=True)
        if self._verbose:
            print_c('HDF5 container has been crated at: <{:}>\n'.format(directory), 'yellow', bold=True)

    def _init_hdf5(self):
        if not self._save:
            print_c(f'/!\\ Saving is disabled', 'red', bold=True)
            return

        if not os.path.isdir(f'./features/{self._session}'):
            # create empty folder
            self._create_folder()

            # create the features hdf5 container with the corresponding conditions
        if not os.path.isfile(f'./features/{self._session}/{self._set_name}_features.hdf5'):
            with h5py.File(f'./features/{self._session}/{self._set_name}_features.hdf5', 'w') as f:
                f.create_dataset('conditions', data=self._conditions)
                f.create_group('features')
                f.create_group('labels')

    @staticmethod
    def _init_set_name(set_name):
        if not set_name:
            return ''
        return set_name.upper()
