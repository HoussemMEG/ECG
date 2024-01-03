import numpy as np

time = np.arange(-0.2, 0.4-1/400, 1/400)
print(time)
print(time.shape)
labels = ['{:.1f}'.format(val) for val in np.arange(-0.2, 0.3, 0.1)]
print(labels)
print(labels.shape)

print('['
      '{:.5f},'.format(np.mean(np.array([0.00016, 0.00015, 0.00015]))),
      '{:.5f},'.format(np.mean(np.array([0.00019, 0.00016, 0.00016]))),
      '{:.5f},'.format(np.mean(np.array([0.00009, 0.00010, 0.00012]))),
      '{:.5f},'.format(np.mean(np.array([0.00016, 0.00014, 0.00015]))),
      '{:.5f},'.format(np.mean(np.array([0.00009, 0.00010, 0.00009]))),
      '{:.5f},'.format(np.mean(np.array([0.00012, 0.00011, 0.00012]))),
      '{:.5f},'.format(np.mean(np.array([0.00018, 0.00017, 0.00014]))),
      '{:.5f},'.format(np.mean(np.array([0.00021, 0.00020, 0.00020]))),
      '{:.5f},'.format(np.mean(np.array([0.00023, 0.00025, 0.00022]))),
      '{:.5f},'.format(np.mean(np.array([0.00032, 0.00032, 0.00030]))),
      '{:.5f},'.format(np.mean(np.array([0.00033, 0.00034, 0.00032]))),
      '{:.5f} '.format(np.mean(np.array([0.00027, 0.00025, 0.00027]))),
      '],')

# import numpy as np
#
# a = [
# 33652 ,
# 37096 ,
# 37102 ,
# 36370 ,
# 39142]
#
# a = np.array(a) / 12 / 50
# print(np.mean(a), np.std(a))

# import numpy as np
# from featuregen import DFG
# import matplotlib.pyplot as plt
# import matplotlib
# from scipy import signal
#
# matplotlib.use('QT5agg')
# plt.rcParams["font.family"] = "monospace"
# fs = 300
# f = 7
# t = np.linspace(0, 1-(1/fs), fs, endpoint=True)
# y = np.sin(2 * np.pi * f * t)
# y = signal.square(2 * np.pi * f * t)
# # y = np.concatenate((np.zeros(11), y))
#
# feature_gen = DFG(method='LARS',
#                   f_sampling=fs,
#                   version=1,
#                   find_alpha=False,
#                   alpha=[0.00000001],  # 0.0008
#                   model_freq=[f, 2*f], # np.linspace(0.1, fs/2, 50),
#                   normalize=True,
#                   damping=None,  # (under-damped 0.008 / over-damped 0.09)
#                   fit_path=True, ols_fit=True,
#                   fast=True,
#                   selection=[],  # 0.02
#                   selection_alpha=None,
#                   omit=None,  # omit either 'x0' or 'u' from y_hat computation
#                   plot=(False, True), show=True, fig_name="fig name", save_fig=True,
#                   verbose=True)
# feature_gen.generate(y)
#
# plt.show()
#

# import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch
# import numpy as np
# import matplotlib
# from operator import sub
#
# matplotlib.use('QT5agg')
# plt.rcParams["font.family"] = "monospace"
#
#
# fig, ax = plt.subplots(figsize=(10, 5))
#
# def get_aspect(ax):
#     # Total figure size
#     figW, figH = ax.get_figure().get_size_inches()
#     # Axis size on figure
#     _, _, w, h = ax.get_position().bounds
#     # Ratio of display units
#     disp_ratio = (figH * h) / (figW * w)
#     # Ratio of data units
#     # Negative over negative because of the order of subtraction
#     data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
#
#     return data_ratio / disp_ratio
# def patch_figure(ax: matplotlib.pyplot.Axes):
#     for s in ax.spines:
#         ax.spines[s].set_visible(False)
#     p_bbox = FancyBboxPatch(xy=(0, 0), width=1, height=1,
#                             boxstyle="round, rounding_size=0.02, pad=0",
#                             ec="#1A1A1A", fc="white", clip_on=False, lw=1.2,
#                             mutation_aspect=get_aspect(ax),
#                             transform=ax.transAxes)
#     ax.add_patch(p_bbox)
#     ax.patch = p_bbox
#     return ax
#
# ax = patch_figure(ax)
# ax.set_xlabel('x label')
# t = np.linspace(0, 2 * np.pi, 1000)
# ax.plot(np.sin(3 * t), np.sin(4 * t))  # drawing some curve
#
#
# plt.show()

# import numpy as np
# from sklearn.svm import LinearSVC
# # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#
# # clf = LinearDiscriminantAnalysis()
# clf = LinearSVC()
#
# rep = 100000
# n = 4
# dim = 2
# mem = []
# for i in range(rep):
#     a = np.random.randint(0, 2, size=(n, dim))
#     # b = np.random.randint(0, 2, n)
#     b = np.array([0, 0, 1, 1])
#     print(i, '\n', a, b)
#     clf.fit(a, b)
#     score = clf.score(a, b)
#     mem.append(score)
#     print('score:', score, '\n')
#
# mem = np.array(mem)
# print(np.mean(mem))

# import numpy as np
# import itertools
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
#
# # clf = LinearSVC()
# clf = SVC()
# n = 8
#
# m = int(n/2)
# # lst = np.array(list(map(list, itertools.product([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], repeat=n))))
# lst = np.array(list(map(list, itertools.product([0, 1, 2, 3, 4, 5, 6, 7], repeat=n))))
# # lst = np.array(list(map(list, itertools.product([0, 1, 2, 3], repeat=n))))
# # lst = np.array(list(map(list, itertools.product([0, 1], repeat=n))))
# print(lst, '\n\n')
# # lst = np.array([[0, 3, 1, 2]])
# y = [*[0]*m, *[1]*m]
#
#
# mem = []
# for i, x in enumerate(lst):
#     gg = x[np.newaxis, :].T
#     a = [[] for _ in range(n) ]
#     for j, elem in enumerate(gg):
#         if elem == [0]:
#             a[j] = [0, 0, 0, 0]
#         elif elem == [1]:
#             a[j] = [0, 0, 1, 0]
#         elif elem == [2]:
#             a[j] = [0, 1, 0, 0]
#         elif elem == [3]:
#             a[j] = [0, 1, 1, 0]
#         elif elem == [4]:
#             a[j] = [0, 0, 0, 1]
#         elif elem == [5]:
#             a[j] = [0, 0, 1, 1]
#         elif elem == [6]:
#             a[j] = [0, 1, 0, 1]
#         elif elem == [7]:
#             a[j] = [0, 1, 1, 1]
#         elif elem == [8]:
#             a[j] = [1, 0, 0, 0]
#         elif elem == [9]:
#             a[j] = [1, 0, 1, 0]
#         elif elem == [10]:
#             a[j] = [1, 1, 0, 0]
#         elif elem == [11]:
#             a[j] = [1, 1, 1, 0]
#         elif elem == [12]:
#             a[j] = [1, 0, 0, 1]
#         elif elem == [13]:
#             a[j] = [1, 0, 1, 1]
#         elif elem == [14]:
#             a[j] = [1, 1, 0, 1]
#         elif elem == [15]:
#             a[j] = [1, 1, 1, 1]
#     gg = np.array(a)
#     # print(gg)
#     clf.fit(gg, y)
#     score = clf.score(gg, y)
#     print(i, *zip(x, y), score)
#     mem.append(score)
# print('Final score', np.mean(mem))