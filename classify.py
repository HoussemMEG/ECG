import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tabulate import tabulate

from config_classifier import *
from plot import Plotter
from reader import Reader
from utils import update_table, prints

reader = Reader(batch_size=batch_size * len(condition))
pca = PCA(n_components=pca_n_comp)
plotter = Plotter()

for x_train, y_train in reader.read_hdf5(session, 'train', condition, random=True):
    print(x_train.shape)
    print(y_train)
    targets = targets if targets else range(x_train.shape[1])  # noqa
    parsimony = parsimony if parsimony else parameters['parsimony']  # noqa
    non_zero_masks = []
    ###########
    score_mem = []
    table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]
    for pars_idx, pars in enumerate(parsimony):
        x = x_train[:, :, :, pars_idx].reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
        if use_non_zero_mask:
            mask = np.any((x != 0), axis=0)
            non_zero_masks.append(mask)
            x = x[:, mask]
        print(x.shape)
        clf[pars_idx].fit(x, y_train)
        score = clf[pars_idx].score(x, y_train) * 100
        score_mem.append(score)
        if score >= max_:  # noqa
            max_ = score
    update_table(table, parsimony, score_mem, highlight_above=87)
    prints(table, 1, targets)
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
    break
print('max training value {:.1f}'.format(max_))


# del x_train, y_train
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
    score_mem = []
    table = [['Parsimony (%)'], ['Train (%)'], ['Remark']]
    for pars_idx, pars in enumerate(parsimony):
        x = x_val[:, :, :, pars_idx].reshape((x_val.shape[0], x_val.shape[1] * x_val.shape[2]))
        if use_non_zero_mask:
            x = x[:, non_zero_masks[pars_idx]]  # noqa
        score = clf[pars_idx].score(x, y_val) * 100
        score_mem.append(score)
    update_table(table, parsimony, score_mem, highlight_above=87)
    prints(table, 1, targets)
    break
plt.show()
