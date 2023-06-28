import time
import warnings
from math import pi
from typing import Union, List, Tuple, Optional
import random

import numpy as np
from scipy.linalg import block_diag, expm
from sklearn import linear_model
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import normalize

import utils
from plot import Plotter

np.set_printoptions(precision=2, suppress=True)


class DFG:
    """
    ------ Dynamical Feature Generator ------
    The main contribution of our paper is in this class.
    Given a signal {y} with the shape: (n_point, n_target)
    This class returns a set of dynamical features (VMS) with different level of parsimony for each target of {y}
    The dynamical features have the shape: List(n_target)(n_features = n_freq * n_point, n_path)
    n_path varies from target to target, representing different level of parsimony
    The selection of the level of parsimony is feasible (for more information check the method _select())
    """
    random.seed(42)

    def __init__(self,
                 method: Optional[str]='LARS',
                 version=1,
                 f_sampling=None,
                 model_freq=None,
                 damping=None,
                 alpha=None,
                 normalize=True, ols_fit=True, max_iter=2000, fit_path=True,
                 selection=None, fast=False,
                 selection_alpha=None,
                 plot=(False, False), show=True, save_fig=False, fig_name='figure_name',
                 **kwargs):
        """
        Nomenclature:
            path: refers to the solutions with varying level of sparsity. The level of sparsity decreases at each
                  iteration so it also decreases along the path indexes.
            target: refers to the signal "y" that we want to fit with our model. We can have multiple targets at the
                    same time to boost the performances (not used in main).
            coef_path: refers to the Virtual Modal Stimuli (VMS) computed at each iteration (for a varying level of
                       sparsity). In some place it contain "U" and "x0" values then theses values are separated inside
                       the method self._compute_y_hat(...) such that coef_path will only contain the values of "U".
            x0_path: x0 values for each iteration.
            alpha_path: values of alpha corresponding to each solution resulting at each iteration.
            intercept_path: intercept values for each iteration.
            residue_path: Loss function values along the path.
            y_hat: resulting fitted signal for each level of sparsity. For version 1 and 2 you can omit the
                   contribution of either 'u' or 'x0' to the predicted signal.


        :param: method: Method choice, available methods {'lasso', 'lars', None}
        :param: version: (int) can only be: [0, 1, 2].
                        version 0: consider x0=0 by putting phi_1 = 0.
                        version 1: The one presented in the paper.
                        version 2: instead of finding x0 using a LASSO-Lars, x0 is fitted using a least square before
                                    we start fitting the U. This will generate a full x0 with a sparse U. The number
                                    of used frequencies in the model impact this version a lot.
                                        contact the corresponding author for more information.
        :param: f_sampling: Sampling frequency of your signal (must be constant) and the same for all the signal targets
        :param: model_freq: List containing the frequencies within our generative system
        :param: damping: (float) Instead of using un-damped oscillators, we can add a damping factor. The experimental
                        values are 0.008 for under-damped system and 0.09 for over-damped system. Your value should be
                        in between. If None no damping is applied.
        :param: alpha: List of constant that multiplies the penalty term for each target. If scalar is provided, all
                      regressor will have the same alpha. Ignored when find_alpha = True
                      for more information check sklearn.linear_model.LassoLars or sklearn.linear_model.Lasso
        :param: normalize: If True, the regressor _phi will be normalized before regression by subtracting the mean and
                          dividing by the l2-norm. Set to False only if you know what you are doing
        :param: ols_fit: Parameters ignored for 'Lasso' method. I recommend you to let it's value True
                        (more information in the method _complete_fit())
        :param: max_iter: Maximum number of iterations to perform.
        :param: fit_path: If True the full path (different level of parsimony) is stored in the coef_path attribute.
                         If you compute the solution for a large problem or many targets, setting fit_path to False
                            will lead to a speedup, especially with a small alpha
        :param: selection (List[int], List[float], None): Only available for the method 'Lars' and when fit_path=True.
                          If None, no selection performed and the entire coefficient path is returned (the entire
                             different level of parsimony).
                          If List[int], returns the coefficient given the path indexes specified.
                          If List[float], returns the coefficient given their percentage position in n_path
                             (EX. 0.5 gives back the coefficient in the middle of the path)
                          All out of bound indexes are ignored.
        :param: selection_alpha (List[float], None): Select the solutions that the level of parsimony alpha is the
                               neighbor of the given values. To have an idea, the selection is
                                    np.argmin(residue + selection_alpha * np.count_nonzero(coef_path, axis=0))
        :param: fast (bool): Ignored for 'Lass' method and available only if fit_path=True and ols_fit=True.
                            If True ols_fit is applied to selected coefficients only (reduce drastically the
                            computation time).
                            Plot coefficient along with their frequencies, residue path and the coefficient number
                            is disabled.
                            (more information in the method _fast_select())
        :param: plot: True if you plots are desired (fit comparison, coefficient along with their frequencies,
                     residue path and the coefficient number <the residue value while varying the number
                     of parameter/level or parsimony>)
        :param: show: True in order to show the plotted figures, ignores when plot=False
        :param: save_fig: True in order to save the plotted figures, ignored when plot=False
        :param: fig_name: Figure name
        :param: kwargs: merging_weight: of the matrices phi_1 and phi_2, 0 < float < 1.
        """

        # Algorithm parameters
        self._method = str(method).upper()
        self.alpha = alpha
        self._find_alpha, self._best_alpha = kwargs.get('find_alpha', False), []
        self._max_iter = max_iter
        self._normalize = normalize
        self._fit_path = fit_path
        if type(selection) == float:
            selection = np.arange(selection, 1 + selection, selection)
        self._selection: Union[List[int], List[float], None] = list(selection) if not selection_alpha else None
        self._selection_alpha: Union[List[int], List[float], None] = selection_alpha
        self._idx_selection = []  # (n_target, (n_alpha_select or n_selection))
        self._fast = fast
        self._version = version
        self._ols_fit = ols_fit  # ordinary least square regressor
        self._ols = linear_model.LinearRegression(fit_intercept=True,
                                                  n_jobs=1)  # keep n_jobs = 1 if multi processing is enabled

        # Method's internal parameters
        self._freq = np.linspace(3, 45, 40, endpoint=True) if model_freq is None else np.array(model_freq)
        self._n_freq = len(self._freq)
        self._damping = 0 if damping is None else damping  # damping val: (under-damped 0.008 / over-damped 0.09)
        self._omit = kwargs.get('omit', None) if self._version != 0 else None
        self._A = None
        self._B = None
        self._C = None
        self._C_type = kwargs.get('C_type', ['scaled', 'unscaled'][0])
        self._phi = None
        self._phi_offset = None
        self._phi_scale = None
        self._merging_weight = kwargs.get('merging_weight', 0.5)

        # Signal parameters
        self._fs = f_sampling
        self._n_point = None
        self._n_target = None

        # Result containers
        # list containers because their value's dimension is inconsistent (each target has it's own path size)
        self.coef_path: List[np.ndarray] = []
        self.x0_path: List[Optional[np.ndarray]] = []
        self.alpha_path: List[np.ndarray] = []
        self.intercept_path: List[np.ndarray] = []
        self.residue_path: List[np.ndarray] = []
        self.y_hat: List[np.ndarray] = []

        # Plotting parameters
        self._plot = plot
        self._show = show
        self._save = save_fig
        self._fig_name = fig_name

        # Parameters saving
        self.parameters = None

        self._verbose = kwargs.get('verbose', False)

    @utils.execution_time
    def generate(self, y: np.ndarray) -> Tuple[Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
        """
        Main method that generates the features given a signal {y}

        :param: y: signal with the shape: (n_point, n_target)
        :return: coef_path: The VMS in a tuple (U, x0) with each element has the shape:
                            List(n_target)(n_features = n_freq * n_point, n_path)
                            n_path represents the different level of parsimony (see _select() for more information)
        """
        # algorithm checks and verifications
        y = self._check(y)

        # retrieve y shape
        self._n_point = y.shape[0]
        self._n_target = y.shape[1]

        # return if method is None
        if self._method is None:
            return None, None

        # create phi matrices and init the regressor, this step is run once at the start
        if self._phi is None:
            self._create_system()
            self._init_regressor()

        # for each target generate the SDF
        for target_idx in range(self._n_target):
            self._generate(self._lasso[target_idx], y[:, [target_idx]])

        # plot the data
        self._plot_(y)

        self._apply_selection()

        to_return = (self.coef_path[:], self.x0_path[:])  # return all the n_targets
        self._save_parameters()  # save the running parameters
        self._reset()  # reset the parameters for next run
        return to_return

    def _generate(self, lasso, y: np.ndarray):
        """
        Method in charge of fitting the regressor to the data and for parsing the result (extracting and formatting
        the coefficient path, alpha path and the residue path)

        :param: lasso: inst that performs the feature generation depending on the chosen method
                      (sklearn.linear_model.LassoLars or sklearn.linear_model.Lasso)
        :param: y: signal with the shape: (n_point, 1)
        :return: coef_path, x0_path, alpha_path, intercept_path, residue_path, y_hat_path.
        """

        # Fitting the data
        if self._version == 2:
            y_ = y[:]
            y = self._phi_tilda @ y

        with warnings.catch_warnings(record=True) as warning:
            lasso.fit(X=self._phi, y=y)
            if any([issubclass(warn.category, ConvergenceWarning) for warn in warning]):
                utils.print_c('\t\t/!\\ Convergence warning', 'red')

        # Coefficients parsing and OLS fit
        if self._method == 'LARS':
            # getting the coefficients
            if self._fit_path:
                coef_path = lasso.coef_path_
            else:
                coef_path = lasso.coef_.T

            # selection part
            selection = self._select(n_path=coef_path.shape[1])
            if self._fast:  # select directly the important parsimony levels to speed up the calculation
                coef_path = coef_path[:, selection]
            if self._ols_fit:  # ordinary least square to complete fitting
                coef_path = self._complete_fit(y, coef_path)

            # rescaling back the coefficients
            coef_path = coef_path / self._phi_scale

        elif self._method == 'LASSO':
            coef_path = lasso.coef_[:, np.newaxis] / self._phi_scale  # / self._phi_scale was added

        # intercept values
        intercept_path = np.empty(coef_path.shape[1])
        for path_idx in range(coef_path.shape[1]):
            intercept_path[path_idx] = np.mean(y) - np.dot(self._phi_offset, coef_path[:, path_idx].T)

        # y_hat values
        if self._version == 2:
            y = y_
        y_hat_path, coef_path, x0_path = self._compute_y_hat(coef_path=coef_path, intercept_path=intercept_path, y=y)

        # alpha values
        alpha_path = lasso.alphas_ if self._method == 'LARS' else np.array([lasso.alpha])
        if self._fast and self._method == 'LARS':
            alpha_path = alpha_path[selection]

        # residue values
        residue_path = self._compute_residue(y=y, y_hat_path=y_hat_path, coef_path=coef_path,
                                             alpha_path=alpha_path, norm='l0')

        # Appending the new calculated values for each target to the overall parameters container
        self.coef_path.append(coef_path)
        self.x0_path.append(x0_path)
        self.alpha_path.append(alpha_path)
        self.intercept_path.append(intercept_path)
        self.residue_path.append(residue_path)
        self.y_hat.append(y_hat_path)

    def _select(self, n_path):
        """ old fast select comments
        Light weight selection method (only available when fast=True and ols_fit=True)
        in order to make the computation of OLS fast
        (the OLS is adjusted for only the coefficients with the selected
        level of parsimony instead of all the different parsimony levels available)

        (See _select() for more information)

        :param: n_path: number paths (level of parsimony) in the coefficients
        :return: selection: the index of the selected coefficients according to their level of parsimony
        """
        """
        Perform path (level of parsimony) selection (only available for the method 'Lars' and when fit_path=True)
        The lowest path_index represent coefficients with the highest parsimony (few number of coefficients)
            but the lowest match to the signal {y}
        While the highest path_index represents coefficients with the lowest parsimony (several number of coefficients)
            but the highest match to the signal {y}

        If fast=True no selection is performed and no data is stored.

        Selection (List[int], List[float], None)
          If None, no selection performed and the entire coefficient path is returned (the entire
             different level of parsimony).
          If List[int], returns the coefficient given the path indexes specified.
          If List[float], returns the coefficient given their percentage position in n_path
             (EX. [0.0, 0.5, 1.0] returns the first, the middle and the last coefficient of the path)
          All out of bound indexes are ignored.
          /!\\ The list element of selection should be all from the same type

        :return:/
        """
        if not self._fit_path or not self._selection:
            selection = range(n_path)
            self._idx_selection.append(selection)
            return selection

        if all(isinstance(x, int) for x in self._selection):
            selection = sorted(list(set([x for x in self._selection if x < n_path])))

        elif all(isinstance(x, float) for x in self._selection):
            selection = sorted(list(set([int(np.round(x * (n_path - 1))) for x in self._selection
                                         if (x >= 0.0) and (x <= 1.0)])))

            if len(selection) != len(self._selection):
                repeat = len(self._selection) // len(selection)
                diff = len(self._selection) - repeat * len(selection)
                to_duplicate = random.sample(selection, diff)
                selection = utils.duplicate_element(selection, to_duplicate, repeat=repeat - 1)
                utils.print_c('\t\t/!\\ selection did not found enough sparsity levels to select, {:} solutions are'
                              ' duplicated randomly'.format(diff), 'red')
            # to select indexes in the neighbor of a specified selection value
            # (selection sparsity parameters should be modified manually in the saving file to work)
            # if len(selection) == 1:
            #     n_idx = 10
            #     selection_ = selection[0]
            #     selection = [selection_ + (i - n_idx) for i in range(3 * n_idx + 1)]
            #     utils.print_c('\t\t/!\\ selecting <{:}> values around <{:}>'.format(2 * n_idx + 1, selection_), 'red')

        else:
            raise ValueError('All selection values must have the same type and be int or float only, given type {:}'
                             .format([type(x) for x in self._selection]))

        self._idx_selection.append(selection)
        return selection

    def _apply_selection(self):
        if self._fast:
            return

        for target_idx in range(self._n_target):
            selection = self._idx_selection[target_idx]
            self.coef_path[target_idx] = self.coef_path[target_idx][:, selection]
            self.alpha_path[target_idx] = self.alpha_path[target_idx][selection]
            self.intercept_path[target_idx] = self.intercept_path[target_idx][selection]
            self.residue_path[target_idx] = self.residue_path[target_idx][selection]
            self.y_hat[target_idx] = self.y_hat[target_idx][:, selection]
            if self._version == 1 or self._version == 2:
                self.x0_path[target_idx] = self.x0_path[target_idx][:, selection]

    def _create_system(self):
        """
        Method that creates the system (taking into account the indicated frequencies)
        This method is called once during the instance's lifetime

        :return: /
        """
        if self._verbose:
            print('\tFeature generation : {:}'.format(self._method))
            print('\t\tVersion: {:}'.format(self._version))
            print('\t\talpha = {:}'.format(self.alpha))
            print('\t\tDamping = {:}'.format(self._damping))
            print('\t\tSelection = {:}'.format(np.array(self._selection)))
            print('\t\tOLS fit: {:}'.format(self._ols_fit))
            print('\t\tFast: {:}'.format(self._fast))
            print("\t\tModel frequencies:\n", self._freq)
        tic = time.time()  # time it
        w = 2 * pi * np.array(self._freq)
        ts = 1 / self._fs
        _A = block_diag(*[[[0, 1],
                           [-value ** 2 - (self._damping * value) ** 2, - 2 * self._damping * value]] for value in w])
        self._A = expm(_A * ts)  # High order exp approx (more stable)
        self._B = block_diag(*np.array([[[0], [ts]]] * self._n_freq))
        if self._C_type == 'unscaled':
            self._C = np.array([1, 0] * self._n_freq)
        elif self._C_type == 'scaled':
            self._C = np.array([[self._fs * 2 * pi * f, 0] for f in self._freq]).flatten()

        # phi matrix creation
        if self._version == 0:  # This version does not include x0
            self._phi = np.zeros([self._n_point, self._n_point * self._n_freq])
            cache = np.zeros([self._n_point, self._n_freq])
            for i in range(self._n_point):
                cache[i, :] = self._C @ np.linalg.matrix_power(self._A, i) @ self._B
                for j in range(self._n_point):
                    if j <= i:
                        self._phi[i, j * self._n_freq:(j + 1) * self._n_freq] = cache[i - j, :]
            self._phi_unscaled = self._phi[:]

            # Normalizing
            self._phi_scale = np.ones([self._n_point * self._n_freq, 1])
            if self._normalize:
                self._phi, self._phi_scale = normalize(self._phi, axis=0, copy=True, return_norm=True)
                self._phi_scale = self._phi_scale[:, np.newaxis]

        else:  # The matrix phi including x0 has the following form: [phi_1 phi_2]*[x0 U].T = [y]
            phi_1 = np.zeros([self._n_point, 2 * self._n_freq])
            phi_2 = np.zeros([self._n_point - 1, (self._n_point - 1) * self._n_freq])
            zero = np.zeros([1, (self._n_point - 1) * self._n_freq])

            # Cache and phi_1 matrix creation
            for i in range(self._n_point):
                phi_1[i, :] = self._C @ np.linalg.matrix_power(self._A, i)

            # phi_2 matrix creation
            for i in range(self._n_point - 1):
                for j in range(self._n_point - 1):
                    if j <= i:
                        phi_2[i, j * self._n_freq:(j + 1) * self._n_freq] = phi_1[i - j, :] @ self._B
            phi_2 = np.vstack((zero, phi_2))

            if self._version == 1:  # x0 and U have a weighted likelihood to be picked
                # scaling and merging the two phi matrices
                self._phi = np.hstack((phi_1, phi_2))
                self._phi_scale = np.ones([(self._n_point + 1) * self._n_freq, 1])
                if self._normalize:
                    phi_1, phi_1_scale = normalize(phi_1, axis=0, copy=True, return_norm=True)
                    phi_2, phi_2_scale = normalize(phi_2, axis=0, copy=True, return_norm=True)
                    self._phi = np.hstack((phi_1, phi_2))
                    if self._merging_weight != 0.5:  # if merging weight is different from 0.5 merge them traditionally
                        self._phi = np.hstack((self._merging_weight * phi_1, (1 - self._merging_weight) * phi_2))
                        phi_1_scale /= self._merging_weight
                        phi_2_scale /= (1 - self._merging_weight)
                    self._phi_scale = np.concatenate((phi_1_scale, phi_2_scale))[:, np.newaxis]
                self._phi_unscaled = self._phi[:] * self._phi_scale.T

            if self._version == 2:  # x0 is firstly fitted with an OLS then U is computed following a lasso(lars)
                self._phi_1 = phi_1
                self._phi_1_plus = np.linalg.pinv(phi_1)
                self._phi_tilda = (np.eye(self._n_point) - phi_1 @ self._phi_1_plus)
                self._phi = self._phi_tilda @ phi_2
                self._phi_unscaled = self._phi[:]
                self._phi_scale = np.ones([(self._n_point - 1) * self._n_freq, 1])
                if self._normalize:
                    self._phi, self._phi_scale = normalize(self._phi, axis=0, copy=True, return_norm=True)
                    self._phi_scale = self._phi_scale[:, np.newaxis]

        # phi offset to compute the intercept value
        self._phi_offset = np.average(self._phi_unscaled, axis=0)
        if self._verbose:
            print("\t\tPhi shape = {:}".format(self._phi.shape))
            utils.print_c("\t\tCreation time: {:.3} (s)".format(time.time() - tic), 'blue')

    def _init_regressor(self):
        """
        Method that initializes and returns a linear regression model based on the specified method and parameters.

        :return: the initialized linear regression model
        """
        lasso_ = []

        # shape verification
        if isinstance(self.alpha, list) and len(self.alpha) != self._n_target:
            raise ValueError("{:} alphas were provided while {:} is needed.".format(len(self.alpha), self._n_target))

        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            self.alpha = [self.alpha] * self._n_target

        if self._method == 'LARS':
            for alpha in self.alpha:
                lasso_.append(linear_model.LassoLars(alpha=alpha, max_iter=self._max_iter, fit_path=self._fit_path,
                                                     fit_intercept=True, normalize=False, verbose=0))
        elif self._method == 'LASSO':
            for alpha in self.alpha:
                lasso_.append(linear_model.Lasso(alpha=alpha, tol=1e-4, max_iter=self._max_iter, fit_intercept=True))
        self._lasso = lasso_

    def _compute_y_hat(self, coef_path, intercept_path, y):
        """
        Method to compute the predicted y_hat given the coefficients computed
        This method is also used to separate properly between the control coefficients U and the x0 coefficients
        in the case where version == {1, 2}

        :param: coef_path: predicted coefficients, shape: (n_features, n_path)
        :param: intercept_path: intercept values along the path, shape: (n_path,)
        :param: y: input signal given to the algorithm shape: (n_point, 1) (support only 1 target)
        :return: y_hat_path: predicted y given our model shape: (n_point, n_path)
        :return: coef_path: predicted U coefficients, shape: (n_features, n_path)
        :return: x0_path: predicted x0 coefficients, shape: (2 * n_freq, n_path)
        """
        if self._version == 0:
            y_hat_path = self._phi_unscaled @ coef_path + intercept_path
            return y_hat_path, coef_path, None

        elif self._version == 1:
            # omit the contribution of x0 or U
            if self._omit in ['x0', 'X0']:
                coef_path[0:2 * self._n_freq, :] = 0  # to put x0 = 0
            elif self._omit in ['u', 'U']:
                coef_path[2 * self._n_freq:, :] = 0  # to put u = 0

            y_hat_path = self._phi_unscaled @ coef_path + intercept_path
            x0_path = coef_path[0:2 * self._n_freq, :]
            coef_path = coef_path[2 * self._n_freq:, :]

        elif self._version == 2:
            x0_path = self._phi_1_plus @ (y - self._phi_unscaled @ coef_path)
            # omit the contribution of x0 or U
            y_hat_x0 = self._phi_1 @ x0_path if self._omit not in ['x0', 'X0'] else 0
            y_hat_U = self._phi_unscaled @ coef_path if self._omit not in ['u', 'U'] else 0

            # the prediction y_hat is the sum of the effect of each activation (x0 and U)
            y_hat_path = y_hat_x0 + y_hat_U + intercept_path

        # Scaling x0 to compare it to U
        x0_path *= np.array([[self._fs * 2 * pi * f, self._fs] for f in self._freq]).flatten()[:, np.newaxis]
        return y_hat_path, coef_path, x0_path

    def _compute_residue(self, y, y_hat_path, coef_path, alpha_path, norm='l0'):
        """
        Method that compute the residue or the loss function along the path of coefficients
                L = 1/(2*N) * sum((y-y_hat)**2) + {alpha * {|U|_0, |U|_1}, 0}

        :param: y: input signal given to the algorithm shape: (n_point, 1) (support only 1 target)
        :param: y_hat_path: predicted y given our model shape: (n_point, n_path)
        :param: coef_path: predicted coefficients, shape: (n_features, n_path)
        :param: alpha_path: alpha values along the path, shape: (n_path,)
        :param: norm: type of norm used in the loss function can be either 'l0' or 'l1' or no norm (None)
        :return: residue_path, shape: (n_path,)
        """
        if not self._fit_path or self._method != 'LARS':
            return None

        if self._fast and self._selection_alpha:
            raise ValueError('Alpha selection {:} activated while fast option being {:}'.format(self._selection_alpha,
                                                                                                self._fast))

        sq_error = 1 / (2 * self._n_point) * np.linalg.norm((y - y_hat_path), ord=2, axis=0)

        self._find_best_alpha(sq_error, y, alpha_path, threshold=0.01)

        if self._selection_alpha:
            select = []
            for alpha in self._selection_alpha:
                if norm.lower() == 'l0':
                    select.append(np.argmin(sq_error + alpha * np.count_nonzero(coef_path, axis=0)))
                elif norm.lower() == 'l1':
                    select.append(np.argmin(sq_error + alpha * np.abs(coef_path, axis=0)))
            self._idx_selection.append(select)

        if norm is None:
            return sq_error
        if norm.lower() == 'l0':
            residue_path = sq_error + alpha_path * np.count_nonzero(coef_path, axis=0)
        elif norm.lower() == 'l1':
            residue_path = sq_error + alpha_path * np.linalg.norm(coef_path, ord=1, axis=0)
        return residue_path

    def _find_best_alpha(self, sq_error, y, alpha_path, threshold=0.01):
        if not self._find_alpha:
            return

        relative_error = sq_error * 2 * self._n_point / np.linalg.norm(y, ord=2)

        if relative_error[-1] > threshold:
            utils.print_c('\t\t/!\\ consider lowering the value of alpha, solution under {:} % threshold could'
                          ' not be found'.format(threshold * 100), 'red')

        idx = np.argmin(np.abs(relative_error - threshold))
        best_alpha = alpha_path[idx]
        self._best_alpha.append(best_alpha)

    def _complete_fit(self, y, coef_path):
        """
        Method that finish fit given the activated coefficient
        It performs an Ordinary Least Square (OLS) taking into account the non-zero coefficients only.
        This method works only with the method='Lars' else ignored

        :param: y: signal with the shape: (n_point, 1)
        :param: coef_path: coefficient part calculated at that step, with the shape: (n_features, n_path)
        :return: coef_path: same coefficient with a better fit
        """
        # Maybe I should add a check if a parameter switched the sign to remove it (such as LASSO-Lars modification)
        if self._fit_path:
            for path_idx in range(0, coef_path.shape[1]):
                no_zero_mask = np.where(coef_path[:, path_idx] != 0)[0]
                if not len(no_zero_mask) == 0:
                    self._ols.fit(X=self._phi[:, no_zero_mask], y=y)
                    coef_path[no_zero_mask, path_idx] = self._ols.coef_[0]
        else:
            no_zero_mask = np.where(coef_path != 0)[0]
            self._ols.fit(X=self._phi[:, no_zero_mask], y=y)
            coef_path[no_zero_mask, :] = self._ols.coef_.T
        return coef_path

    def _plot_(self, y):
        """
        Method that handles all the plots, each target and each path iteration will have its own plot.
        /!\\ to plot fast should be set to False since we need to have all the path solutions to plot them.

        Coefficient number and residue: <the residue value while varying the number of parameter/level or parsimony>
        if only available for the method='Lars' and when fit_path=True.

        Match comparison: plotting the initial signal {y} and {y_hat} with different level of parsimony and for
        each target.

        Coefficient: showing the frequencies that have been activated and their corresponding activation instant.

        :param: y: signal to be plotted, shape: (n_point, n_target)
        :return: /
        """
        if not any(self._plot):
            return

        for target_idx in range(self._n_target):
            fig_name = self._fig_name.split('.')[0] + '_' + str(target_idx)
            selection = self._idx_selection[target_idx] if self._idx_selection else None
            if self._plot[0] and self._fit_path and self._method == 'LARS' and not self._fast:
                Plotter.plot_alpha(self.alpha_path[target_idx], self.coef_path[target_idx],
                                   self.residue_path[target_idx], fig_name, self._show, self._save, selection)

            if self._plot[1]:
                Plotter.plot_control(x=self.coef_path[target_idx],
                                     y=y[:, target_idx], y_hat=self.y_hat[target_idx],
                                     freq=self._freq, cmap='blue_red',  # 'blue_red', 'black_white'
                                     fig_name=fig_name, save=self._save, show=self._show)

    def _reset(self):
        """
        Reset the result to use the generate method several times

        :return: /
        """
        self.coef_path: List[np.ndarray] = []
        self.x0_path: List[Optional[np.ndarray]] = []
        self.alpha_path: List[np.ndarray] = []
        self.intercept_path: List[np.ndarray] = []
        self.residue_path: List[np.ndarray] = []
        self.y_hat: List[np.ndarray] = []
        self._idx_selection = []

    def _check(self, y):
        """
        Check if the right class parameters are provided and if y has the appropriate shape

        :param: y: given signal his shape must be: (n_point, n_target) or (n_point,)
        :return: y with shape: (n_point, n_target) or (n_point, 1)
        """
        # Error values
        if self._method not in ['LARS', 'LASSO', 'NONE']:
            raise ValueError("Unrecognized {:} method, available methods : ['LARS', 'LASSO']".format(self._method))
        if self._method in ['LARS', 'LASSO']:
            if self.alpha is None:
                raise ValueError("Please provide the value of the parameter <alpha>")

        if not isinstance(self._version, int):
            raise ValueError("Method versions accepted {:} but got {:} instead".format([0, 1, 2], self._version))

        if self._selection_alpha and self._fast:
            raise ValueError("Alpha sparsity selection while fast is {:}".format(self._fast))

        # warnings and reminders
        if not self._selection_alpha and self._selection and not self._fast and self._verbose:
            utils.print_c(r'/!\ Fast is not enabled', 'red')
        if self._fast and self._plot[0] and self._verbose:
            utils.print_c(r'/!\ Cannot plot alpha values if fast is enabled', 'red')
        if not self._ols_fit and self._verbose:
            utils.print_c(r'/!\ OLS fit not enabled', 'red')
        if self._version == 1 and not self._normalize and self._merging_weight != 0.5 and self._verbose:
            utils.print_c(r'/!\ Normalize set to {:} while merging_weight is set to {:}, Set Normalize to <True> for'
                          r' the intended behaviour.'.format(self._normalize, self._merging_weight))

        # set the best parameter for alpha finding
        if self._find_alpha:
            self.alpha = 8e-6
            self._fast = False  # in order to fit all the sparsity levels
            self._selection = None
            self._selection_alpha = None
            if self._verbose and self._phi is None:
                utils.print_c('/!\\ Looking for the best value of alpha, for a better precision some parameters has been'
                              ' set to:\n   <alpha = {:}>, <fast = {:}>, <selection = {:}>, <selection_alpha = {:}> '
                              .format(self.alpha, self._fast, self._selection, self._selection_alpha), 'red', bold=True)

        # check y dimension
        if y.ndim == 1:
            return y[:, np.newaxis]
        elif y.ndim > 2:
            raise ValueError('y must be a 2 dimensional array')
        return y

    def _save_parameters(self):
        """
        Saving the session parameters (for reproducibility)
        /!\\ be careful only the first running parameters are saved

        :return: dict containing the parameters to save
        """
        if self.parameters is None:
            self.parameters = {'method': self._method,
                               'alpha': self.alpha,
                               'version': self._version,
                               'merging_weight': self._merging_weight if self._version == 1 else None,
                               'max_iter': self._max_iter,
                               'normalize': self._normalize,
                               'fit_path': self._fit_path,
                               'selection': self._selection,
                               'selection_alpha': self._selection_alpha,
                               'find_alpha': self._find_alpha,
                               'ols_fit': self._ols_fit,
                               'model_freq': self._freq.tolist(),
                               'n_freq': self._n_freq,
                               'damping': self._damping,
                               'sampling_frequency': self._fs,
                               'n_point': self._n_point,
                               'n_features': self._n_point * self._n_freq if self._version == 0
                               else (self._n_point - 1) * self._n_freq,
                               'fig_name': self._fig_name,
                               'C_type': self._C_type}

    @property
    def selection(self) -> List[np.ndarray]:
        return self.coef_path

    @selection.setter
    def selection(self, selection: Union[List[int], List[float], None]):
        if self._fast:
            raise ValueError('No selection modification is allowed when fast=True, '
                             'please set fast=False to change the selection mid execution.')
        self._selection = selection
        self._select()

    def __del__(self):
        if self._find_alpha and self._best_alpha:
            for target in range(self._n_target):
                # utils.print_c('best alpha of channel {:<2} is {:}'
                #               .format(target, self._best_alpha[target::self._n_target]), 'yellow')
                utils.print_c('best alpha of channel {:<2} is {:.5f}'
                              .format(target, np.mean(self._best_alpha[target::self._n_target])), 'yellow')
            print('')
