import time
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

import utils

np.set_printoptions(precision=2)

session = ["2023-04-06 66;66 200 of each small window",
           "2023-04-06 66;66 100 of each",
           "2023-04-06 66;66 only healthy and RBBB",
           "2023-04-06 66;66"][1]

condition = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST', 'HEALTHY']
# condition = ['RBBB', 'HEALTHY']

# clf = [LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, covariance_estimator=None) for _ in range(12 * 50)]
# clf = [QuadraticDiscriminantAnalysis() for _ in range(12 * 50)]
clf = [RandomForestClassifier(max_depth=10, max_leaf_nodes=100, warm_start=False, random_state=42, n_jobs=-1) for _ in range(12 * 50)]

batch_size = 200
pca_n_comp = 10
do_pca = False
use_non_zero_mask = True

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
