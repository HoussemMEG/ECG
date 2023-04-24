import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

clf = LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=[0.5, 0.5], n_components=None, tol=0.0001)

n = 200
balance = 0.5
n_repeat = 1000  # 20 * 64 * 30
n_dim = 30

max_val = 0
for i in tqdm(range(n_repeat)):
    SZ = np.random.uniform(-1, 1, size=(int(balance * n), n_dim))
    y_SZ = np.ones((len(SZ),))
    CTL = np.random.uniform(-1, 1, size=(int((1 - balance) * n), n_dim))
    y_CTL = np.zeros((len(CTL),))
    X = np.concatenate((SZ, CTL))
    if X.ndim < 2:
        X = X[:, np.newaxis]
    y = np.concatenate((y_SZ, y_CTL))
    clf.fit(X, y)
    score = clf.score(X, y) * 100
    print(f'{i = }\t{score = :.2f}')
    if score > max_val:
        max_val = score
print('\t\tMax score obtained: {:.2f}'.format(max_val))