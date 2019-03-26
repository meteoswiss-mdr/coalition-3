""" [COALITION3] From the statistic in 2D dataframe, predict TRT Rank at
    different lead times with XGB models."""

from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import datetime

import numpy as np
import pandas as pd

from coalition3.inout.paths import path_creator_vararr, path_creator_UV_disparr

## =============================================================================
## FUNCTIONS:

## ============================================================================

def perform_prob_matching(value,test_TRT_rank,diff_prob_match):
    idx = (np.abs(test_TRT_rank - value)).idxmin()
    return diff_prob_match[idx]

## Make prediction in operational context:
def predict_TRT_Rank(cfg_set):
    t1 = datetime.datetime.now()
    str_addon = "probability matched " if probability_matching else ""
    print("Predict %sTRT Ranks for future time steps" % str_addon)

    ## Get xarray dataset from tmp/ directory:
    filename_df   = os.path.join(cfg_set["tmp_output_path"],"%s%s" % \
                              (cfg_set["t0_str"],"_stat_pixcount_df_t0diff.h5"))
    filename_pred = os.path.join(cfg_set["tmp_output_path"],"%s%s" % \
                                (cfg_set["t0_str"],"_prediction.h5"))
    #with open(filename_df, "rb") as stat_df_file: stat_df = pickle.load(stat_df_file)
    stat_df = pd.read_hdf(filename_df, "df")
    TRT_Rank_df = pd.DataFrame(stat_df["TRT_Rank|0"].copy())

    with open("/opt/users/jmz/coalition-3/models/XGB_model_allfeat_ls.pkl", "rb") as feat_ls_file:
        feat_ls = pickle.load(feat_ls_file)
    with open("/opt/users/jmz/coalition-3/models/XGB_model_750feat_ls.pkl", "rb") as pred_ls_file:
        pred_ls = pickle.load(pred_ls_file)
    if probability_matching:
        with open("/opt/users/jmz/coalition-3/models/TRT_Rank_obs_pred.pkl", "rb") as TRT_Rank_obs_pred_file:
            TRT_Rank_obs_pred = pickle.load(TRT_Rank_obs_pred_file)

    for dt_i, pred_dt in enumerate(np.arange(5,50,5)):
        top_features = pd.DataFrame.from_dict(feat_ls[dt_i].get_booster().get_score(importance_type='gain'),
                                              orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                              ascending=False)
        TRT_Rank_pred = stat_df["TRT_Rank|0"] + pred_ls[dt_i].predict(stat_df[top_features.index[:750]])
        TRT_Rank_df["TRT_Rank_pred|%i" % pred_dt] = TRT_Rank_pred

        if cfg_set["probability_matching"]:
            test_TRT_rank   = TRT_Rank_obs_pred["TRT_Rank_pred|%i" % pred_dt]
            diff_prob_match = TRT_Rank_obs_pred["TRT_Rank_pred_PM|%i" % pred_dt] - test_TRT_rank
            TRT_Rank_df["TRT_Rank_pred_PM|%i" % pred_dt] = TRT_Rank_df["TRT_Rank_pred|%i" % pred_dt] + \
                                                           [perform_prob_matching(value,test_TRT_rank,diff_prob_match) for value in TRT_Rank_pred.values]

    TRT_Rank_df.to_hdf(filename_pred,"pred",mode="w")
    t2 = datetime.datetime.now()
    print("  Elapsed time for predicting TRT Ranks: "+str(t2-t1)+"\n")











