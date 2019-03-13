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
def get_X_col(colname, del_list=None):
    ## No future TRT observations in input:
    use_col = ("TRT" not in colname or \
               "-" in colname or "|0" in colname) and \
              ("TRT_Rank_diff" not in colname)
    if del_list is not None:
        use_col = use_col and (colname not in del_list)
    return(use_col)
    
def get_radar_t0_vars(TRT_Rank_vars = None):
    radar_vars  = ["RZC","BZC","LZC","MZC","EZC15","EZC20",
                   "EZC45","EZC50","CZC"]
    radar_stats = ["SUM","MEAN","STDDEV","MIN","PERC01","PERC05",
                   "PERC25","PERC50","PERC75","PERC95","PERC99","MAX"]
    vars_stat  = [var+"_stat|0|" for var in radar_vars]
    vars_stat += [var+"_stat_nonmin|0|" for var in radar_vars]
    vars_stat += [var+"_pixc|0|" for var in radar_vars]
    vars_stat += [var+"_pixc_nonmin|0|" for var in radar_vars]
    radar_t0_vars = [var+stat for var in vars_stat+vars_nonmin_stat for stat in radar_stats]
    radar_t0_vars += ["CZC_lt57dBZ|0|SUM","RANKr","RANK","ET15","ET15m",
                   "ET45","ET45m","VIL","area"]
    if TRT_Rank_vars is not None:
        radar_t0_vars += TRT_Rank_vars
    return radar_t0_vars
    
    
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
                    check_for_nans=True,
                    verbose=False):

    """ This function is used to prepare the full 2D training dataframe into 
        objects which can be used for the model training and testing.
        
    Parameters:
    -----------
    
    df_tds : pandas dataframe
        Full training dataframe with all the features (roughly 10100), including
        all TRT Ranks of the future time steps.
        
    del_TRTeqZero_tpred : boolean
        If true, samples where the TRT Ranks drops (close) to zero at the prediction
        time pred_dt. This is the default, such that the models are not confused
        by the TRT Rank jump caused by the (dis)apperance of pixels with EchoTop > 45dBZ.
        Parameter 'pred_dt' has to be provided. As minimum TRT Rank, 0.15 at prediction
        time delta 0.15 is used as default (see parameter 'TRTRank_gap_thresh').
        
    split_Xy : boolean
        If true, split the dataframe into a dataframe X with only predictors (up to t0)
        and a dataframe y with only predictants (TRT Rank at prediction time pred_dt).
        
    split_Xy_traintest : boolean
        If true, split the predictor/predictant dataframes X & y further into training
        and testig dataframes X_train/X_test & y_train/y_test. By default, training and
        testing splits are done such that no TRT Cell (according to the TRT ID) is both
        in the training and the testing dataset (see parameter 'traintest_TRTcell_split').
        Furthermore, the training dataframes are 4x larger than the testing dataframes
        (see parameter 'X_test_size')
        
    X_normalise : boolean
        If true, normalise the predictor dataframes X_train and X_test (subtract mean and
        devide by standard deviation). This is necessary before training an ANN.
        
    pred_dt : int
        Prediction time delta (e.g. 30 for 30min). This has to be provided, since the number
        of samples where the TRT Rank drops to zero changes depending on the lead time
        (see parameter 'del_TRTeqZero_tpred').
        
    TRTRank_gap_thresh : float
        Threshold below which samples are deleted if the TRT Rank falls at prediction time
        delta below this threshold.
        
    TRTRankt0_bound : list of two floats
        Model bounds in terms of TRT Ranks at t0. E.g. by setting it to [1.2, 2.3], only
        samples are selected where the TRT Rank at t0 is between these two model bounds.
    
    X_feature_sel : list of strings
        For the predictor dataframe X, only these features are selected. Default is "all",
        such that all features are selected.
        If X_feature_sel=="no_radar_t0", then all radar variables at t0 are not considered
        for the predictor dataframe X.
        
    X_test_size : float
        Float between 0 and 1, stating the fraction of samples going into the test
        dataframes X_test and y_test.
        
    traintest_TRTcell_split : bool
        If true, when splitting the predictor dataframe X and the predictant dataframe y
        into training and testing dataframes, it is assured that no TRT cell both occurs
        in the training and the testing dataframes. If false, samples a selected randomly.
        
    check_for_nans : bool
        If true, check for nans and delete the respective rows.
    """
                    
    ## Get step when results should be returned:
    ret_step = return_step([split_Xy,split_Xy_traintest])
    
    ## Delete any nan-values
    if check_for_nans:
        df_tds = df_tds.dropna(0,'any')

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

    ## In case a training and testing dataframe seperated by TRT cell ID
    ## should be returned, already select which ID fall in training & testing
    if split_Xy_traintest and traintest_TRTcell_split:
        if verbose:
            print("  Select TRT IDs for training/testing split")
        np.random.seed(seed=42)
        TRT_ID = np.array([dti[13:] for dti in df_tds.index])
        TRT_ID_test = np.random.choice(np.unique(TRT_ID),
                                       int(X_test_size*len(np.unique(TRT_ID))),
                                       replace=False)
        df_tds["Testing"] = False
        df_tds["Testing"].iloc[np.where(np.in1d(TRT_ID, TRT_ID_test))[0]] = True
    
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
        elif X_feature_sel=="no_radar_t0":
            list_TRT_Rank_vars = [var for var in df_tds.columns if "TRT_Rank|" in var]
            del_vars = get_radar_t0_vars(list_TRT_Rank_vars)
            X = df_tds[[Xcol for Xcol in df_tds.columns if get_X_col(Xcol,del_vars)]]
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
            X_train = X.loc[~X["Testing"],:]
            X_train = X_train.drop(labels="Testing", axis=1)
            X_test  = X.loc[ X["Testing"],:]
            X_test  = X_test.drop(labels="Testing", axis=1)
            y_train = y.loc[~X["Testing"]]
            y_test  = y.loc[ X["Testing"]]
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
