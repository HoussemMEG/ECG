import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.decomposition import PCA

from reader import Reader
from utils import print_c, read_parameters, convolution2d

from plot import Plotter
import matplotlib.pyplot as plt

np.set_printoptions(precision=2)


class Classifier:
    def __init__(self, session, selection, clf, condition=None, metric=None, random=True):
        # Parameters
        self._metric = metric.lower() if metric else 'accuracy'
        self.session = session
        self._random = random
        self._condition = condition if condition else []
        self._selection = selection
        self._parameter = read_parameters(session)
        self._clf_choice = clf
        self._gg = None

        # Misc
        self._n_leads = 0
        self._n_pars = 0

        # Containers
        self._clf = []


    def _classify(self, x, y, mode):
        ## testing new idea
        reader = Reader(batch_size=100)
        data_healthy = np.zeros((1, 12, self._parameter['n_features'], 1))
        for x_healthy, _ in reader.read_hdf5(session, 'validation', ['HEALTHY'], random=False):
            data_healthy += np.mean(x_healthy[:, :, :, [-1]], axis=0)
            del x_healthy
        data_healthy /= 1
        ## end of testing

        for exam_i in range(len(x)):
            x[exam_i, :, :, -1] -= data_healthy[0, :, :, 0]

        x_ = self._feature_selection(x)  # (n_exams, n_leads, n_pars, n_features)
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

                    # Feature extraction
                    # x = self._feature_extraction(x)  # not completed

                    # Fit classifier to data
                    if hasattr(clf, 'partial_fit'):
                        clf.partial_fit(x, y, classes=np.unique(y))
                    else:
                        ## testing
                        self._gg = self._set_clf(selection=self._clf_choice.upper())
                        x = self._gg.fit_transform(x, y)
                        Plotter.correlogram(x, y, show=True, save=False)
                        print('training', x.shape)
                        ## end of testing
                        clf.fit(x, y)


                    # Metrics
                    y_pred = clf.predict(x)
                    score = self._score(y, y_pred)
                    print(f'{score = :.2f} \t len {len(self._clf)}')

                if mode == 'evaluation':
                    clf = self._clf[i_lead * self._n_leads + i_pars]
                    ## testing
                    print('testing before fit transform', x.shape)
                    x = self._gg.transform(x)
                    print('testing after fit transform', x.shape)
                    ## end of testing

                    # Metrics
                    y_pred = clf.predict(x)
                    score = self._score(y, y_pred)
                    print(f'{score = :.2f}')
                    pass
        plt.show()
                # Display results

    @staticmethod
    def _feature_extraction(x):
        """
        not completed
        :param x:
        :return:
        """
        # a = np.mean(x, axis=-1)[:, np.newaxis]
        # b = np.max(x, axis=-1)[:, np.newaxis]
        # c = np.argmin(x, axis=-1)[:, np.newaxis]
        # return np.concatenate((a, b, c), axis=-1)
        pca = PCA(n_components=3)
        output = pca.fit_transform(x)
        print('feature extraction output', output.shape)
        return output
        # return np.max(x, axis=-1)[:, np.newaxis]
        # return np.argmax(x, axis=-1)[:, np.newaxis]
        # pass


    @staticmethod
    def _set_clf(selection):
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
        if selection == 'LDA':
            return LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, covariance_estimator=None,
                                              n_components=1)
        elif selection == 'QDA':
            return QuadraticDiscriminantAnalysis()
        elif selection == 'RF':
            return RandomForestClassifier(max_depth=8, max_leaf_nodes=20, warm_start=False, random_state=42, n_jobs=-1)
        elif selection == 'SGD':
            return SGDClassifier()
        elif selection == 'PA':
            return PassiveAggressiveClassifier(n_jobs=-1)


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

            ## testing new idea
            # for exam_i in range(len(temp)):
            #     temp[exam_i, 0, 0, :, :] = convolution2d(temp[exam_i, 0, 0, :, :])
            ## end of testing

            if any(self._selection['time']):
                t_min = self._selection['time'][0] if self._selection['time'][0] is not None else 0
                t_max = self._selection['time'][1] if self._selection['time'][1] is not None else temp.shape[3]
                temp = temp[:, :, :, t_min:t_max, ...]

            if any(self._selection['frequency']):
                f_min = self._selection['frequency'][0] if self._selection['frequency'][0] is not None else 0
                f_max = self._selection['frequency'][1] if self._selection['frequency'][1] is not None else temp.shape[4]
                if isinstance(f_min, float):
                    f_min = np.argmin(self._parameter['model_freq'] - f_min)
                if isinstance(f_max, float):
                    f_max = np.argmin(self._parameter['model_freq'] - f_max)
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
        reader = Reader(batch_size=batch_size * len(self._condition))

        mode_ = 'learning' if mode.lower() == 'train' else 'evaluation'
        # Read the <mode> data and perform the core classification in learning mode
        for x, y in reader.read_hdf5(self.session, mode, self._condition, random=self._random, verbose=False):
            self._classify(x, y, mode=mode_)

    def _score(self, y_true, y_pred):
        """
        Calculate the score for a classification task based on the specified metric.

        :param y_true: True labels for the input samples, shape (n_samples,).
        :param y_pred: Predicted labels for the input samples, shape (n_samples,)

        :return: Score for the specified metric, multiplied by 100 to convert to a percentage.
        :raises ValueError: if the specified metric is not one of ['accuracy', 'f1'].
        """
        # List of available metrics
        all_metrics = ['accuracy', 'f1']

        # Check if the specified metric is valid
        if self._metric not in all_metrics:
            raise ValueError(f"Metric <{self._metric}> not found, only {all_metrics} are available")

        # Calculate the score based on the specified metric
        if self._metric == 'accuracy':
            return accuracy_score(y_true, y_pred) * 100
        elif self._metric == 'f1':
            return f1_score(y_true, y_pred) * 100


selection = {'lead': [7],  # 8
             'parsimony': [-1],  # -1
             'time': (55, 70),  # (55, 70)
             'frequency': (9, 35),  # (0, 35)
             'merge_pars':False,
             'merge_lead':False}

# condition = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']
condition = ['SB', 'HEALTHY']

session = "2023-04-06 66;66 visualisation 100 70 freq"
classifier = Classifier(session, selection, 'LDA', condition)
classifier.classify('train', batch_size=100)
classifier.classify('validation', batch_size=100)
# classifier.classify('test', batch_size=15)
