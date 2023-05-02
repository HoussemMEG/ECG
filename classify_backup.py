import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tabulate import tabulate

from config_classifier import *
from plot import Plotter
from reader import Reader
from utils import update_table, prints

import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import utils

np.set_printoptions(precision=2)

session = ["2023-04-06 66;66 200 of each small window",
           "2023-04-06 66;66 100 of each",
           "2023-04-06 66;66 only healthy and RBBB",
           "2023-04-06 66;66",
           "2023-04-06 66;66 visualisation 100 70 freq",
           "2023-04-06 66;66 visualisation 500 35 freq short window"][-1]

# condition = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']
condition = ['SB', 'HEALTHY']

# clf = [LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, covariance_estimator=None) for _ in range(12 * 50)]
# clf = [QuadraticDiscriminantAnalysis() for _ in range(12 * 50)]
# clf = [RandomForestClassifier(max_depth=8, max_leaf_nodes=20, warm_start=False, random_state=42, n_jobs=-1) for _ in range(12 * 50)]
clf = [SGDClassifier() for _ in range(12 * 50)]

batch_size = 100
pca_n_comp = 5
do_pca = False
use_non_zero_mask = False

parameters = utils.read_parameters(session)
targets = []  # choose specific targets
parsimony = []  # choose specific paths

max_ = 0

# Plot correlogram parameters
show = False and do_pca
save = False and do_pca

targets = [val-1 for val in targets]
"""
# all_conditions ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']
"""


reader = Reader(batch_size=batch_size * len(condition))
pca = PCA(n_components=pca_n_comp)
plotter = Plotter()

for x_train, y_train in reader.read_hdf5(session, 'train', condition, random=True):
    x_train = x_train[:, 7, :, -1]
    x_train = np.array(np.split(x_train, 119, axis=-1)).transpose((1, 0, 2))
    x_train = x_train[:, 65:78, 0:7]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    # x_train = np.mean(x_train, axis=1)[:, np.newaxis]
    # x_train = np.argmax(x_train, axis=1)[:, np.newaxis]
    # x_train = pca.fit_transform(x_train)
    print(x_train.shape)
    print(y_train)
    # x = pca.fit_transform(x_train.reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3])))
    # clf[0].fit(x_train, y_train)
    clf[0].partial_fit(x_train, y_train, np.unique(y_train))
    print('train score', clf[0].score(x_train, y_train) * 100)
    # targets = targets if targets else range(x_train.shape[1])  # noqa
    # parsimony = parsimony if parsimony else parameters['parsimony']  # noqa
    # non_zero_masks = []
    # ###########
    # score_mem = []
    # table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]
    # for pars_idx, pars in enumerate(parsimony):
    #     x = x_train[:, :, :, pars_idx].reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
    #     if use_non_zero_mask:
    #         mask = np.any((x != 0), axis=0)
    #         non_zero_masks.append(mask)
    #         x = x[:, mask]
    #     print(x.shape)
    #     clf[pars_idx].fit(x, y_train)
    #     score = clf[pars_idx].score(x, y_train) * 100
    #     score_mem.append(score)
    #     if score >= max_:  # noqa
    #         max_ = score
    # update_table(table, parsimony, score_mem, highlight_above=87)
    # prints(table, 1, targets)
    #########
    # for target in targets:
    #     score_mem = []
    #     table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]
    #
    #     # for pars_idx, pars in enumerate(parsimony):
    #     #     x = pca.fit_transform(x_train[:, target, :, pars_idx]) if do_pca else x_train[:, target, :, pars_idx]
    #     #     # print(np.any((x != 0), axis=0))
    #     #     # print(np.any((x != 0), axis=0).shape)
    #     #     if use_non_zero_mask:
    #     #         mask = np.any((x != 0), axis=0)
    #     #         non_zero_masks.append(mask)
    #     #         x = x[:, mask]
    #     #     Plotter.correlogram(x, y_train, show=show, save=save)
    #     #     clf[target * len(targets) + pars_idx].fit(x, y_train)
    #     #
    #     #     # metric return score
    #     #     score = clf[target * len(targets) + pars_idx].score(x, y_train) * 100
    #     #     score_mem.append(score)
    #     #     # print('lead {:}\t parsimony {:} %\t score {:.1f} %\t n_component: {:}'.format(target + 1, pars * 2, score, 100))
    #     #     if score >= max_:  # noqa
    #     #         max_ = score
    #     update_table(table, parsimony, score_mem, highlight_above=87)
    #     prints(table, 1, targets)
    # break
print('max training value {:.1f}'.format(max_))

# for x_val, y_val in reader.read_hdf5(session, 'validation', condition, random=True):
#     target = 5 - 1
#     pars_idx = 42 - 1
#     x = pca.fit_transform(x_val[:, target, :, pars_idx]) if do_pca else x_val[:, target, :, pars_idx]
#     if use_non_zero_mask:
#         x = x[:, non_zero_masks[pars_idx]]  # noqa
#     score = clf[target * len(targets) + pars_idx].score(x, y_val) * 100
#     print('validation score', score)

print('\n\n\nVALIDATION')
del x_train, y_train
for x_val, y_val in reader.read_hdf5(session, 'validation', condition, random=True):
    # x = pca.transform(x_val.reshape((x_val.shape[0], x_val.shape[1] * x_val.shape[2] * x_val.shape[3])))
    x_val = x_val[:, 7, :, -1]
    x_val = np.array(np.split(x_val, 239, axis=-1)).transpose((1, 0, 2))
    x_val = x_val[:, 65:78, 0:7]
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
    x_val = np.mean(x_val, axis=1)[:, np.newaxis]
    # x_val = np.argmax(x_val, axis=1)[:, np.newaxis]
    # x_val = pca.transform(x_val)
    print('validation score', clf[0].score(x_val, y_val) * 100)
    # score_mem = []
    # table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]
    # for pars_idx, pars in enumerate(parsimony):
    #     x = x_val[:, :, :, pars_idx].reshape((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
    #     if use_non_zero_mask:
    #         x = x[:, non_zero_masks[pars_idx]]  # noqa
    #     score = clf[pars_idx].score(x, y_val) * 100
    #     score_mem.append(score)
    # update_table(table, parsimony, score_mem, highlight_above=87)
    # prints(table, 1, targets)
    # break
plt.show()
