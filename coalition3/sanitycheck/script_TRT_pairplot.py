# coding: utf-8
from __future__ import division
from __future__ import print_function

import os
import sys

import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import datetime as dt
import dask.dataframe as dd
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import coalition3.inout.paths as pth
from coalition3.visualisation.TRTcells import contour_of_2dHist

def pairgrid_heatmap(x, y, **kws):
    cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
    plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe",user_argv_path)
df = pd.read_hdf(path_to_df,key="df")
df_nonnan = df.dropna(0,'any')

VIL_scal  = df_nonnan[["LZC_stat|0|MAX"]]
ME_scal   = df_nonnan[["CZC_stat|0|MAX"]]
ET45_scal = df_nonnan[["EZC45_stat_nonmin|0|PERC50"]]
A55_scal  = df_nonnan[["CZC_lt57dBZ|0|SUM"]]
VIL_scal  = VIL_scal.where(VIL_scal<65.,65.)   ## Max  VIL:       65 kg m-2
ME_scal   = ME_scal.where(ME_scal>45.,45.)     ## Min MaxEcho:    45 dBZ
ME_scal   = ME_scal.where(ME_scal<57.,57.)     ## Max MaxEcho:    57 dBZ
ET45_scal = ET45_scal.where(ET45_scal<10.,10.) ## Max EchoTop:    10 km
A55_scal  = A55_scal.where(A55_scal<40,40)     ## Max pix >57dBZ: 40

## Scale variables to values between 0 and 4:
VIL_scal  = VIL_scal/65.*4
ET45_scal = ET45_scal/10.*4
ME_scal   = (ME_scal-45.)/12.*4
A55_scal  = A55_scal.astype(np.float32,copy=False)/40.*4

## Calculate TRT rank:
TRT_Rank = (2.*VIL_scal.values+2*ET45_scal.values+ME_scal.values+2.*A55_scal.values)/7.
TRT_Rank = TRT_Rank[:,0]
df_all = pd.concat([pd.DataFrame.from_dict({"TRT Rank":TRT_Rank}).set_index(VIL_scal.index,drop=False),ET45_scal,VIL_scal,ME_scal],axis=1)
sns.pairplot(df_all)
plt.show()

sns.set_style("white")
g = sns.PairGrid(df_all)
g = g.map_diag(plt.hist,bins=20)
g.map_offdiag(pairgrid_heatmap, bins=20, norm=LogNorm())


