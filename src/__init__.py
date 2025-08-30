import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator, ScalarFormatter

from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('retina')
del backend_inline

from matplotlib.font_manager import fontManager
fontManager.addfont('src/lmroman10.otf')
del fontManager

plt.rcdefaults()
plt.rcParams['font.family'] = 'Latin Modern Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 8
plt.rcParams['lines.linewidth'] = 1
plt.rcParams['lines.markersize'] = 3
plt.rcParams['lines.markeredgecolor'] = 'none'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['figure.figsize'] = (8.5 / 2.54, 0.618 * 8.5 / 2.54)
plt.rcParams['figure.dpi'] = 192
# plt.rcParams['axes.prop_cycle'] = plt.matplotlib.rcsetup.cycler(
#     'color', ['tab:purple', 'tab:green', 'tab:blue', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
# )

plt.rcParams['axes.linewidth']=0.5
plt.rcParams['xtick.major.size'] = 4.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.major.width'] = 0.5
plt.rcParams['xtick.minor.width'] = 0.5
plt.rcParams['xtick.minor.visible'] = 'True'
plt.rcParams['ytick.major.size'] = 4.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.major.width'] = 0.5
plt.rcParams['ytick.minor.width'] = 0.5
plt.rcParams['ytick.minor.visible'] = 'True'

from math import tau, pi
import numpy as np
from tqdm import tqdm
from time import sleep
import os
from glob import glob

from datetime import datetime
today = datetime.strftime(datetime.today(), '%Y%m%d')
del datetime


from .core import *
from .estimator import *


# Controller only works on Windows.
import platform
if platform.system() == 'Windows':
    from .controller import *

