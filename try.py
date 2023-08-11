# import numpy as np
#
# print('['
#       '{:.5f},'.format(np.mean(np.array([0.00030, 0.00026, 0.00023, 0.00031, 0.00028]))),
#       '{:.5f},'.format(np.mean(np.array([0.00032, 0.00034, 0.00034, 0.00031, 0.00036]))),
#       '{:.5f},'.format(np.mean(np.array([0.00021, 0.00024, 0.00020, 0.00026, 0.00023]))),
#       '{:.5f},'.format(np.mean(np.array([0.00030, 0.00027, 0.00028, 0.00027, 0.00030]))),
#       '{:.5f},'.format(np.mean(np.array([0.00019, 0.00017, 0.00014, 0.00023, 0.00018]))),
#       '{:.5f},'.format(np.mean(np.array([0.00025, 0.00026, 0.00024, 0.00023, 0.00028]))),
#       '{:.5f},'.format(np.mean(np.array([0.00035, 0.00037, 0.00035, 0.00035, 0.00033]))),
#       '{:.5f},'.format(np.mean(np.array([0.00042, 0.00046, 0.00045, 0.00043, 0.00039]))),
#       '{:.5f},'.format(np.mean(np.array([0.00048, 0.00047, 0.00050, 0.00052, 0.00048]))),
#       '{:.5f},'.format(np.mean(np.array([0.00051, 0.00055, 0.00065, 0.00055, 0.00060]))),
#       '{:.5f},'.format(np.mean(np.array([0.00057, 0.00057, 0.00061, 0.00062, 0.00054]))),
#       '{:.5f} '.format(np.mean(np.array([0.00048, 0.00045, 0.00049, 0.00050, 0.00048]))),
#       '],')

import numpy as np
from featuregen import DFG
import matplotlib.pyplot as plt
import matplotlib
from scipy import signal

matplotlib.use('QT5agg')
plt.rcParams["font.family"] = "monospace"
fs = 300
f = 7
t = np.linspace(0, 1-(1/fs), fs, endpoint=True)
y = np.sin(2 * np.pi * f * t)
y = signal.square(2 * np.pi * f * t)
# y = np.concatenate((np.zeros(11), y))

feature_gen = DFG(method='LARS',
                  f_sampling=fs,
                  version=1,
                  find_alpha=False,
                  alpha=[0.00000001],  # 0.0008
                  model_freq=[f, 2*f], # np.linspace(0.1, fs/2, 50),
                  normalize=True,
                  damping=None,  # (under-damped 0.008 / over-damped 0.09)
                  fit_path=True, ols_fit=True,
                  fast=True,
                  selection=[],  # 0.02
                  selection_alpha=None,
                  omit=None,  # omit either 'x0' or 'u' from y_hat computation
                  plot=(False, True), show=True, fig_name="fig name", save_fig=True,
                  verbose=True)
feature_gen.generate(y)

plt.show()


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
