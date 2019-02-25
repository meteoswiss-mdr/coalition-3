# coding: utf-8
""" [COALITION3] This script contains code for the selection of features using XGBoost"""
    
## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd

import coalition3.inout.paths as pth
import coalition3.inout.readconfig as cfg
import coalition3.statlearn.feature as feat
    
## ============================================================================
## Get config info:
cfg_tds = cfg.get_config_info_tds()
cfg_op, __, __ = cfg.get_config_info_op()

## Open pandas training dataframe:
user_argv_path = sys.argv[1] if len(sys.argv)==2 else None
path_to_df = pth.file_path_reader("pandas training dataframe (nonnan)",user_argv_path)
model_path = pth.file_path_reader("model saving location")
df_nonnan  = pd.read_hdf(path_to_df,key="df_nonnan")

## Delete rows where TRT Rank is close to zero at t0:
print("\nRemove rows where TRT Rank (t0) is close to zero")
df_nonnan_nonzerot0 = df_nonnan.loc[df_nonnan["TRT_Rank|0"]>=0.15]
del(df_nonnan)

## Get feature importance for specified time delta:
ls_pred_dt = [-5]
poss_fcst_steps = np.arange(cfg_op["timestep"],
                            cfg_op["timestep"]*cfg_op["n_integ"],
                            cfg_op["timestep"])
print("\nFor which time delta (%i, %i, .., %i) should the feature selection be made?" % \
      (poss_fcst_steps[0], poss_fcst_steps[1], poss_fcst_steps[-1]))
while not np.all([pred_dt_i in poss_fcst_steps for pred_dt_i in ls_pred_dt]):
    ls_pred_dt = raw_input("  Select forecast step (if several, split with comma): ").split(",")
    ls_pred_dt = [int(pred_dt_i.strip()) for pred_dt_i in ls_pred_dt]
print("  Perform feature selection of lead times %s" % ', '.join([str(pred_dt_i) for pred_dt_i in ls_pred_dt]))

print("\nLoop over different lead times to get feature importance")
for pred_dt in ls_pred_dt:
    feat.get_feature_importance(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path)
    feat.get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path)

## Plot MSE as function of number of features:
feat.plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path)

## Fit model with optimal number of features:
poss_n_feat = np.arange(1,501)
ls_n_feat = [-5]
print("\nFor which number of features (%i, %i, .., %i) should a fit be performed?" % \
      (poss_n_feat[0], poss_n_feat[1], poss_n_feat[-1]))
while not (np.all([n_feat_i in poss_n_feat for n_feat_i in ls_n_feat]) and \
           len(ls_n_feat)==len(ls_pred_dt)):
    ls_n_feat = raw_input("  Select n-feature thresholds (split with comma): ").split(",")
    if len(ls_n_feat)!=len(ls_pred_dt):
        print("    Must choose as many n-feature thresholds as time deltas %s" % ls_pred_dt)
    ls_n_feat = [int(n_feat_i.strip()) for n_feat_i in ls_n_feat]

feat.plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=ls_n_feat)
for pred_dt, n_feat in zip(ls_pred_dt,ls_n_feat):
    feat.plot_pred_vs_obs(df_nonnan_nonzerot0,pred_dt,n_feat,cfg_tds,model_path)












