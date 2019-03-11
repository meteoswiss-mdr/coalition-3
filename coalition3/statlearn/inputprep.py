""" [COALITION3] Functions used for preparing the training dataset before
    feading it into a ML model."""

## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## =============================================================================
## FUNCTIONS:

## Get columns not containing TRT of the future:
def get_X_col(colname):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    return(use_col)
    
## Helper function determining when to return results:
def return_step(bool_input_ls):
    if not np.any(bool_input_ls):
        return "return_df"
    if not bool_input_ls[1]:
        return "Xy"
    else:
        return "Xy_tt"

## Take full training dataset and transform into model input form:
def get_model_input(df_tds,
                    del_TRTeqZero_tpred=True,
                    split_Xy=False,
                    split_Xy_traintest=False,
                    X_normalise=False,
                    pred_dt=None,
                    TRTRank_gap_thresh=0.15,
                    TRTRankt0_bound=[0.0, 4.0],
                    X_feature_sel="all",
                    X_test_size=0.2,
                    traintest_TRTcell_split=True,
                    verbose=False):

    ## Get step when results should be returned:
    ret_step = return_step([split_Xy,split_Xy_traintest])

    ## Delete rows with TRT Rank at t0 outside model bounds:
    if (TRTRankt0_bound!=[0.0, 4.0] and TRTRankt0_bound is not None):
        if verbose:
            print("  Delete rows outside model bounds %s" % TRTRankt0_bound)
        df_tds = df_tds.loc[(df_tds["TRT_Rank|0"]>=TRTRankt0_bound[0]) & \
                            (df_tds["TRT_Rank|0"]<TRTRankt0_bound[1])]

    ## Delete samples with small TRT Ranks at t0
    if verbose:
        print("  Delete rows with TRT Rank (t0) < %.2f" % TRTRank_gap_thresh)
    df_tds = df_tds.loc[df_tds["TRT_Rank|0"]>=TRTRank_gap_thresh]

    ## Delete rows with TRT Rank close to zero at lead time:
    if del_TRTeqZero_tpred:
        if pred_dt is None:
            raise ValueError("Variable 'pred_dt' has to be provided")
        if verbose:
            print("  Delete rows with TRT Rank < %.2f " % TRTRank_gap_thresh+ \
              "at lead time %imin" % pred_dt)
        df_tds = df_tds.loc[df_tds["TRT_Rank|%i" % pred_dt]>=TRTRank_gap_thresh]

    ## Check whether df should be returned now:
    if ret_step == "return_df":
        return df_tds

    ## Split dataframe in X and y
    if split_Xy or split_Xy_traintest:
        if pred_dt is None:
            raise ValueError("Variable 'pred_dt' has to be provided")
        if verbose:
            print("  Split training dataframe into X and y")
        if X_feature_sel=="all":
            X = df_tds[[Xcol for Xcol in df_tds.columns if get_X_col(Xcol)]]
        else:
            X = df_tds[X_feature_sel]
        y = df_tds["TRT_Rank_diff|%i" % pred_dt]
        del(df_tds)

    if ret_step == "Xy":
        return X, y

    ## Split training/testing data:
    if split_Xy_traintest:
        if traintest_TRTcell_split:
            if verbose:
                print("  Split into training/testing data (TRT cell separated)")
            np.random.seed(seed=42)
            TRT_ID = np.array([dti[13:] for dti in X.index])
            TRT_ID_test = np.random.choice(np.unique(TRT_ID), int(X_test_size*len(np.unique(TRT_ID))), replace=False)
            X_train = X.iloc[np.where(~np.in1d(TRT_ID, TRT_ID_test))[0],:]
            y_train = y.iloc[np.where(~np.in1d(TRT_ID, TRT_ID_test))[0]]
            X_test  = X.iloc[np.where( np.in1d(TRT_ID, TRT_ID_test))[0],:]
            y_test  = y.iloc[np.where( np.in1d(TRT_ID, TRT_ID_test))[0]]
        else:
            if verbose:
                print("  Split randomly into training/testing data")
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                test_size=X_test_size,
                                                random_state=42)
        del(X,y)

    ## Normalise input data:
    if X_normalise:
        if verbose:
            print("  Normalise input data X")
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = pd.DataFrame(scaler.transform(X_train),index=X_train.index,
                        columns=X_train.columns).astype(np.float32,copy=False)
        X_test  = pd.DataFrame(scaler.transform(X_test),index=X_test.index,
                        columns=X_test.columns).astype(np.float32,copy=False)
        return X_train, X_test, y_train, y_test, scaler
    else:
        return X_train, X_test, y_train, y_test
