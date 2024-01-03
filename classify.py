import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans, Birch
import tensorflow as tf

from reader import Reader
from utils import print_c, read_parameters, convolution2d

from plot import Plotter
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


class Classifier:
    def __init__(self, session, selection, clf, condition=None, conv2d=None, use_all_data=False, metric=None, random=True,
                 clusters=None):
        # Parameters
        self._metric = metric.lower() if metric else 'accuracy'
        self.session = session
        self._random = random
        self._condition = condition if condition else []
        self._selection = selection
        self._conv2d = conv2d if conv2d else {'do_filter': False}
        self._parameter = read_parameters(session)
        self._clf_choice = clf
        self._use_all_data = use_all_data

        # Misc
        self._n_leads = 0
        self._n_pars = 0
        self._clusters = clusters

        # Containers
        self._clf = []


    def _classify(self, x_, y, mode):
        scores = []

        for i_lead in range(self._n_leads):
            for i_pars in range(self._n_pars):
                x = x_[:, i_lead, i_pars, :]

                if mode == 'learning':
                    # Init classifier
                    if len(self._clf) < self._n_pars * self._n_leads:
                        clf = self._set_clf(selection=self._clf_choice.upper())
                        self._clf.append(clf)
                    else:
                        clf = self._clf[i_lead * self._n_leads + i_pars]

                    # temporary here to test CNN
                    if self._clf_choice == 'CNN':
                        x = np.array(np.split(x, self._selection['time'][1]-self._selection['time'][0], axis=1)).transpose((1, 0, 2))
                        for i in range(len(y)):
                            if y[i] == 6:
                                y[i] = 1
                        temp = np.zeros((y.size, 2))
                        temp[np.arange(y.size), y] = 1
                        y = temp

                    # Feature extraction
                    # x = self._feature_extraction(x)  # not completed

                    # Fit classifier to data
                    if hasattr(clf, 'partial_fit') and not self._use_all_data:
                        print('Partially fitting')
                        clf.partial_fit(x, y)  #, classes=np.unique(y))
                    else:
                        clf.fit(x, y)

                    # Metrics
                    y_pred = clf.predict(x)
                    score = self._score(y, y_pred)
                    print(f'{score = :.2f} \t len {len(self._clf)}')

                if mode == 'evaluation':
                    clf = self._clf[i_lead * self._n_leads + i_pars]

                    # # temporary here to test CNN
                    if self._clf_choice == 'CNN':
                        x = np.array(np.split(x, self._selection['time'][1]-self._selection['time'][0], axis=1)).transpose((1, 0, 2))
                        for i in range(len(y)):
                            if y[i] == 6:
                                y[i] = 1
                        temp = np.zeros((y.size, 2))
                        temp[np.arange(y.size), y] = 1
                        y = temp

                    # Feature extraction
                    # x = self._feature_extraction(x)  # not completed

                    # Metrics
                    y_pred = clf.predict(x)
                    score = self._score(y, y_pred)
                    print(f'{score = :.2f}')
                    pass

                # Display results

    @staticmethod
    def _feature_extraction(x):
        """
        not completed
        :param x:
        :return:
        """
        a = np.mean(x, axis=-1)[:, np.newaxis]
        b = np.max(x, axis=-1)[:, np.newaxis]
        c = np.argmin(x, axis=-1)[:, np.newaxis]
        return np.concatenate((a, b, c), axis=-1)
        pca = PCA(n_components=3)
        output = pca.fit_transform(x)
        print('feature extraction output', output.shape)
        return output


    # @staticmethod
    def _set_clf(self, selection):
        """
        Method to perform the selection of the classifier.
        Available classifiers are:
            'LDA': Linear Discriminant Analysis
            'QDA': Quadratic Discriminant Analysis
            'RF':  Random forest
            'SGD': Stochastic Gradient Descent
            'PA':  Passive Agressive Classifier
        :param selection: 'str' to choose the returned classifier.
        :return: clf
        """
        # param = {'classifier': self._clf_choice}
        # print(f'selection={self._selection},   conv2d={self._conv2d},   classifier={param}')

        if selection == 'LDA':
            return LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, covariance_estimator=None,
                                              n_components=1)
        elif selection == 'QDA':
            return QuadraticDiscriminantAnalysis()
        elif selection == 'RF':
            max_depth = 20
            max_leaf_nodes = 200
            param = {'classifier': self._clf_choice, 'max_depth': max_depth, 'max_leaf_nodes': max_leaf_nodes}
            print(f'selection={self._selection},   conv2d={self._conv2d},   classifier={param}')
            return RandomForestClassifier(max_depth=max_depth, max_leaf_nodes=max_leaf_nodes, warm_start=False, random_state=42, n_jobs=-1)
        elif selection == 'SGD':
            return SGDClassifier()
        elif selection == 'KMEANS':
            batch_size = 1280  # 256 * 5
            return MiniBatchKMeans(n_clusters=self._clusters, batch_size=batch_size)
        elif selection == 'BIRCH':
            return Birch(threshold=0.5, branching_factor=50, n_clusters=self._clusters)
        elif selection == 'PA':
            return PassiveAggressiveClassifier(n_jobs=-1)
        elif selection == 'CNN':
            input_shape = (self._selection['time'][1]-self._selection['time'][0],
                           self._selection['frequency'][1]-self._selection['frequency'][0], 1)
            num_classes = 2

            model = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(num_classes, activation='softmax')])
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            return model
        elif selection == 'ADABOOST':
            n_estimators = 100
            learning_rate = 0.5
            param = {'classifier': 'adaBoost', 'n_estimators': n_estimators, 'lr':learning_rate}
            print(f'selection={self._selection},   conv2d={self._conv2d},   classifier={param}')
            return  AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

    def _feature_selection(self, x):
        """
        Method to perform the feature selection based on the self._selection dictionary.
            Select leads in <selection['lead']> if empty no selection is performed
            Select parsimony in <selection['pars']> if empty no selection is performed
            Select time in <selection['time'] = (t_min, t_max)> if t_min = None, no selection will be performed on
                left side of time. Same goes for t_max
            Select Frequency in <selection['frequency'] = (f_min, f_max)> if f_max = None, no selection will be
                performed on upper frequency. Same goes for f_min. If float is given, select directly, if fload are
                provided select based on model frequency.
            Merge all parsimony levels if <selection['merge_pars'] = True>, else nothing is changed
            Merge all leads if <selection['merge_lead'] = True>, else nothing is changed
        :param x: input data array with shape (n_exams, n_leads, n_features, n_parsimony)
        :return: x: output data after selection, /!\ shape has been changed to (n_exams, n_leads, n_parsimony, n_features)
        """
        x = x.transpose((0, 1, 3, 2))

        # Lead selection
        if self._selection['lead']:
            x = x[:, [*self._selection['lead']], ...]

        # Parsimony level selection
        if self._selection['parsimony']:
            x = x[:, :, [*self._selection['parsimony']], ...]

        # Time or/and frequency selection
        if any(self._selection['time']) or any(self._selection['frequency']):
            # temps shape: (n_exams, n_leads, n_parsimony, time, frequency)
            temp = np.array(np.split(x, self._parameter['n_point'] - 1, axis=-1)).transpose((1, 2, 3, 0, 4))

            ## IDEA of gaussian convolution
            if self._conv2d['do_filter']:
                for exam_i in range(len(temp)):
                    temp[exam_i, 0, 0, :, :] = convolution2d(temp[exam_i, 0, 0, :, :], kernel_type=self._conv2d['type'],
                                                             l=self._conv2d['l'], sigma=self._conv2d['sigma'])

            if any(self._selection['time']):
                t_min, t_max = self._get_time_indexes()
                temp = temp[:, :, :, t_min:t_max, ...]

            if any(self._selection['frequency']):
                f_min, f_max = self._get_freq_indexes()
                temp = temp[..., f_min:f_max]
            x = np.reshape(temp, (temp.shape[0], temp.shape[1], temp.shape[2], temp.shape[3] * temp.shape[4]))

        # Merge parsimony indexes
        if self._selection['merge_pars'] and x.shape[2] > 1:
            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))[:, :, np.newaxis, :]

        # Merge leads
        if self._selection['merge_lead'] and x.shape[1] > 1:
            x = x.transpose((0, 2, 1, 3))
            x = np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))[:, :, np.newaxis, :]
            x = x.transpose((0, 2, 1, 3))

        self._n_leads = x.shape[1]
        self._n_pars = x.shape[2]
        return x


    def classify(self, mode, batch_size=10):
        print_c(f'{mode}', 'blue', bold=True)
        mode_ = 'learning' if (mode.lower() == 'train' or mode.lower() == 'learning') else 'evaluation'

        # Init reader and x, y for the case of use_all_data = True
        reader = Reader(batch_size=batch_size * len(self._condition))
        hdf5_batch_iterator = reader.read_hdf5(self.session, mode, self._condition, random=self._random, verbose=True)
        x, y = self._init_x_y(hdf5_batch_iterator)

        pointer = 0
        for x_batch, y_batch in hdf5_batch_iterator:
            x_batch = self._feature_selection(x_batch)  # (n_exams, n_leads, n_pars, n_features)
            if self._use_all_data:
                x[pointer:pointer + len(x_batch)] = x_batch
                y[pointer:pointer + len(x_batch)] = y_batch
                pointer += len(x_batch)
            else:
                # Regular batch classification
                self._classify(x_batch, y_batch, mode=mode_)

            # x = x_batch
            # y = y_batch
            # break

        # lighted ram load
        del x_batch, y_batch

        # Aggregate all exams in one batch (can only be possible if there is selection or low memory usage)
        if self._use_all_data:
            self._classify(x, y, mode=mode_)

    def _init_x_y(self, reader_hdf5):
        n_exams, n_leads, n_features, n_pars = next(reader_hdf5)
        if self._use_all_data:
            if self._selection['lead']:
                n_leads = len(self._selection['lead'])
            if self._selection['parsimony']:
                n_pars = len(self._selection['parsimony'])

            if any(self._selection['time']) or any(self._selection['frequency']):
                t_min, t_max = self._get_time_indexes()
                f_min, f_max = self._get_freq_indexes()
                n_features = (f_max - f_min) * (t_max - t_min)

            if self._selection['merge_lead'] and n_leads > 1:
                n_features *= n_leads
                n_leads = 1

            if self._selection['merge_pars'] and n_pars > 1:
                n_features *= n_pars
                n_pars = 1

            return np.zeros((n_exams, n_leads, n_pars, n_features)), np.empty((n_exams,), dtype=np.int8)

    def _get_time_indexes(self):
        t_min = self._selection['time'][0] if self._selection['time'][0] is not None else 0
        t_max = self._selection['time'][1] if self._selection['time'][1] is not None else self._parameter['n_point'] - 1
        return t_min, t_max

    def _get_freq_indexes(self):
        f_min = self._selection['frequency'][0] if self._selection['frequency'][0] is not None else 0
        f_max = self._selection['frequency'][1] if self._selection['frequency'][1] is not None else self._parameter['n_freq']
        if isinstance(f_min, float):
            f_min = np.argmin(self._parameter['model_freq'] - f_min)
        if isinstance(f_max, float):
            f_max = np.argmin(self._parameter['model_freq'] - f_max)
        return f_min, f_max

    def _score(self, y_true, y_pred):
        """
        Calculate the score for a classification task based on the specified metric.

        :param y_true: True labels for the input samples, shape (n_samples,).
        :param y_pred: Predicted labels for the input samples, shape (n_samples,)

        :return: Score for the specified metric, multiplied by 100 to convert to a percentage.
        :raises ValueError: if the specified metric is not one of ['accuracy', 'f1'].
        """
        # List of available metrics
        all_metrics = ['accuracy', 'f1', 'precision']

        ## testing CNN
        if self._clf_choice == 'CNN':  #remove one hot encoding
            for i, val in enumerate(y_pred):
                if val[0] > val[1]:
                    y_pred[i][0] = 1
                    y_pred[i][1] = 0
                else:
                    y_pred[i][0] = 0
                    y_pred[i][1] = 1
            y_pred = [np.where(r==1)[0][0] for r in y_pred]
            y_true = [np.where(r==1)[0][0] for r in y_true]

        # Check if the specified metric is valid
        if self._metric not in all_metrics:
            raise ValueError(f"Metric <{self._metric}> not found, only {all_metrics} are available")

        # Calculate the score based on the specified metric
        if self._metric == 'accuracy':
            return accuracy_score(y_true, y_pred) * 100
        elif self._metric == 'f1':
            return f1_score(y_true, y_pred) * 100
        elif self._metric == 'precision':
            return precision_score(y_true, y_pred, average=None) * 100


selection = {'lead': [6],
             'parsimony': [-1],
             'time': (39, 104),
             'frequency': (None, None),
             'merge_pars':False,
             'merge_lead':False}

conv2d = {'do_filter': True,
          'type': 'gaussian',
            'l': 5,
            'sigma': 2}

print_c('lead {:}'.format(selection['lead']), 'yellow', bold=True)
print_c('time {:}'.format(selection['time']), 'yellow', bold=True)
print_c('frequency {:}'.format(selection['frequency']), 'yellow', bold=True)
print_c('do_filter {:}'.format(conv2d['do_filter']), 'yellow', bold=True)


# condition = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']  [0, 1, 2, 3, 4, 5, 6]
condition = ['HEALTHY', 'RBBB']

# session = "2800 per category _ 20 freq _ short window"
# session = "2800 per category _ 35 freq _ long window"
session = "2800(last) per category _ 25 freq _ long window"

classifier = Classifier(session, selection, 'adaboost', condition, conv2d=conv2d, use_all_data=True, random=False,  # 'Birch' 'Kmeans'
                                            clusters=len(condition), metric='accuracy')
# classifier.classify('train', batch_size=100)
classifier.classify('learning', batch_size=100)
# classifier.classify('validation', batch_size=100)
classifier.classify('test', batch_size=100)
