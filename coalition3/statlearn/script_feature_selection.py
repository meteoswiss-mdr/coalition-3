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
print("\nLoading nonnan dataframe into RAM")
df_nonnan  = pd.read_hdf(path_to_df,key="df_nonnan")

## Delete rows where TRT Rank is close to zero at t0:
print("\nRemove rows where TRT Rank (t0) is close to zero")
df_nonnan_nonzerot0 = df_nonnan.loc[df_nonnan["TRT_Rank|0"]>=0.15]
del(df_nonnan)

## Get feature importance for specified time delta:
## Get lead times:
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

## Get model boundaries:
ls_model_bound = []; ls_model_names = []
len_input = 0; keep_asking = True
print("\nShould different features for different TRT Ranks (t0) be fitted? (if not, press 'n')")
while (len_input!=2 and keep_asking):
    ls_input = raw_input("  Please provide model boundaries (split with comma, stop with 'n'): ").split(",")
    if ls_input[0]=='n':
        keep_asking = False
        break
    elif len(ls_input)!=2:
        print("    Please provide TWO numbers")
        continue
    else:
        ls_input = [np.float32(ipt) for ipt in ls_input]
    if (np.min(ls_input)<0 or np.max(ls_input)>4):
        print("    Please provide bounds within TRT Rank range (0 to 4)")
    else:
        ls_model_bound.append(ls_input)
        ls_model_names.append(raw_input("  Please provide model name: "))
if len(ls_model_bound)>0:
    print("  Using model boundaries:") # %s" % ls_model_bound)
    for bound, name in zip(ls_model_bound, ls_model_names):
        print("   Model '%s': %s" % (name, bound))
    use_model_boundaries = True
else:
    print("  Using all samples")
    use_model_boundaries = False
    ls_model_bound.append(None)
    ls_model_names.append("")

print("\nLoop over different lead times to get feature importance")
for pred_dt in ls_pred_dt:
    for bounds, name in zip(ls_model_bound, ls_model_names):
        feat.get_feature_importance(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,
                                    mod_bound=bounds,mod_name=name)
        feat.get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,
                                 mod_bound=bounds,mod_name=name)
        
## Plot MSE as function of number of features:
feat.plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=None,
                          ls_model_names=ls_model_names)

## Fit model with optimal number of features:
poss_n_feat = np.arange(1,501)
ls_n_feat_dt = []
for pred_dt in ls_pred_dt:
    ls_n_feat = [-5]
    if ls_model_names[0]!="":
        model_name = " for model(s) '%s'" % "/".join(ls_model_names)
    else:
        model_name = ""
    print("\nHow many features (%i, %i, .., %i) should a used for time delta %imin?" % \
          (poss_n_feat[0], poss_n_feat[1], poss_n_feat[-1],pred_dt))
    while not (np.all([n_feat_i in poss_n_feat for n_feat_i in ls_n_feat]) and \
               len(ls_n_feat)==len(ls_model_names)):
        ls_n_feat = raw_input("  Select n-feature thresholds%s (split with comma): " % model_name).split(",")
        if len(ls_n_feat)!=len(ls_model_bound):
            print("    Must choose as many %i n-feature thresholds" % (len(ls_model_bound)))
        ls_n_feat = [int(n_feat_i.strip()) for n_feat_i in ls_n_feat]
    print("  The following thresholds were selected for time delta %imin:" % pred_dt)
    for model_name, n_feat in zip(ls_model_names, ls_n_feat):
        print("    %s -> %i" % (model_name,n_feat))
    ls_n_feat_dt.append(ls_n_feat)

ls_n_feat_dt_flat = [item for sublist in ls_n_feat_dt for item in sublist]
feat.plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=ls_n_feat_dt_flat,
                          ls_model_names=ls_model_names)
for i_dt, pred_dt in enumerate(ls_pred_dt):
    feat.plot_pred_vs_obs(df_nonnan_nonzerot0,pred_dt,ls_n_feat_dt[i_dt],cfg_tds,model_path,ls_model_bound,ls_model_names)











