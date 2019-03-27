""" [COALITION3] Functions used for the selection of features at different
    leadtimes."""

## Import packages and define functions:
from __future__ import division
from __future__ import print_function

import os
import sys

import pickle
import sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import datetime as dt
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from pandas.api.types import CategoricalDtype

from scipy import stats
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

import coalition3.statlearn.inputprep as ipt
from coalition3.visualisation.TRTcells import contour_of_2dHist
from coalition3.visualisation.TRTcells import truncate_cmap

## =============================================================================
## FUNCTIONS:

## Get prediction time from user:
def get_pred_dt_ls(input_str, timestep=None,n_integ=None):
    ls_pred_dt = [-5]
    if (timestep is not None and n_integ is not None):
        poss_fcst_steps = np.arange(timestep,
                                    timestep*n_integ,
                                    timestep)
        print("\nFor which time delta (%i, %i, .., %i) should %s be made?" % \
              (poss_fcst_steps[0], poss_fcst_steps[1], poss_fcst_steps[-1], input_str))
    else:
        poss_fcst_steps = np.arange(-100,100)
        print("\nFor which time delta should %s be made?" % input_str)

    while not np.all([pred_dt_i in poss_fcst_steps for pred_dt_i in ls_pred_dt]):
        ls_pred_dt = raw_input("  Select forecast step (if several, split with comma): ").split(",")
        ls_pred_dt = [int(pred_dt_i.strip()) for pred_dt_i in ls_pred_dt]
    print("  Perform %s at lead times %s min" % (input_str, ', '.join([str(pred_dt_i) for pred_dt_i in ls_pred_dt])))
    return ls_pred_dt

## Plotting procedure for feature importance:
def plot_feature_importance(model,X,delta_t,cfg_tds,mod_name):
    sort_ind = np.argsort(model.feature_importances_)[::-1]
    df_featimp = pd.DataFrame(np.array([X.columns[sort_ind[:500]],
                                        model.feature_importances_[sort_ind[:500]]]).T,
                              columns=["Feature","Importance"])
    df_featimp.plot(drawstyle="steps", linewidth=2)
    plt.grid(); plt.ylabel("Feature importance"); plt.xlabel("Features (sorted)")
    plt.title("Feature importance - TRT t+%imin" % delta_t)
    #plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin%s_long.pdf" % (delta_t,mod_name)),orientation="portrait")

    fig = plt.figure(figsize = [10,10])
    ax1 = fig.add_subplot(2,1,1)
    xgb.plot_importance(model,ax1,max_num_features=20,importance_type="weight",
                        title="Feature importance (Weight) - TRT t+%imin" % delta_t)
    ax2 = fig.add_subplot(2,1,2)
    xgb.plot_importance(model,ax2,max_num_features=20,importance_type="gain",
                        title="Feature importance (Gain) - TRT t+%imin" % delta_t,show_values=False)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Feature_importance_%imin%s.pdf" % (delta_t,mod_name)),orientation="portrait")

## Get sample weight for Training dataset:
def calc_sample_weight(TRT_Rank0, TRT_Rank_diff):
    s_weight = np.exp(TRT_Rank0) * np.exp(TRT_Rank0+TRT_Rank_diff)
    return s_weight

## Get feature ranking for the complete dataset:
def get_feature_importance(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,mod_bound=None,
                           mod_name="",delete_RADAR_t0=False,set_log_weight=False,max_n_feat=80000):
    print("Get features for lead time t0 + %imin" % pred_dt, end="")
    if mod_bound is not None:
        if mod_name=="":
            raise ValueError("Model name required")
        else:
            print(" (for %s)" % mod_name)
            mod_name = "_%s" % mod_name
        if len(mod_bound)!=2:
            raise ValueError("Model boundary list must have length 2")
    else:
        print(" (for all samples)")
    sys.stdout.flush()

    ## Check whether model already exists:
    if os.path.exists(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name))):
        use_existing = ""
        while (use_existing!="y" and use_existing!="n"):
            use_existing = raw_input("  Model exists alreay, fit a new one? [y/n] ")
        if use_existing=="n":
            print("  Use existing one, return from this function")
            return

    ## Calculate sample weights for XGB fitting:
    if set_log_weight:
        df_nonnan_nonzerot0["s_weight"] = calc_sample_weight(df_nonnan_nonzerot0["TRT_Rank|0"],
                                                             df_nonnan_nonzerot0["TRT_Rank_diff|%i" % pred_dt])

    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    if delete_RADAR_t0:
        print("  Get predictor matrix X without RADAR variables at t0")
        X_feature_sel = "no_radar_t0"
    else:
        print("  Get predictor matrix X with RADAR variables at t0")
        X_feature_sel = "all"
    X, y = ipt.get_model_input(df_nonnan_nonzerot0, del_TRTeqZero_tpred=True,
            split_Xy=True, pred_dt=pred_dt, TRTRankt0_bound=mod_bound, X_feature_sel=X_feature_sel)
    del(df_nonnan_nonzerot0)
    if len(X)>max_n_feat:
        print("   *** Warning: Dataframe X probably to big to be converted, reduced to %i rows! ***" % max_n_feat)
        X = X.sample(n=max_n_feat,random_state=42)
        y = y.sample(n=max_n_feat,random_state=42)
    #X = X.values
    #X = X.astype(np.float16, order='C', copy=False)

    ## Setup model:
    print("  Setup XGBmodel with max_depth = 6")
    xgb_model = xgb.XGBRegressor(max_depth=6,silent=False,n_jobs=6,nthreads=6)

    ## Calculate sample weights for XGB fitting:
    if set_log_weight:
        s_weights = X["s_weight"].values
        X = X.drop(labels="s_weight", axis=1)
    else:
        s_weights = None

    ## Train model:
    print("  Train XGBmodel")
    d_start = dt.datetime.now()
    xgb_model.fit(X, y, verbose=True, sample_weight=s_weights)
    print("    Elapsed time for XGBoost model fitting: %s" % (dt.datetime.now()-d_start))

    ## Save model to disk:
    print("  Save XGBmodel to disk")
    with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(xgb_model,file,protocol=2)

    ## Plot feature importance:
    print("  Plot feature importance")
    plot_feature_importance(xgb_model,X,pred_dt,cfg_tds,mod_name)

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def plot_feat_source_dt_gainsum(path_xgb, cfg_op, cfg_tds, pred_dt_ls = None):
    print("Plot feature source and time delta importance")
    if pred_dt_ls is None:
        pred_dt_ls  = np.arange(cfg_op["timestep"],cfg_op["n_integ"]*cfg_op["timestep"],cfg_op["timestep"])
        print("  This script assumes that for all future time steps %imin - %imin, " % (pred_dt_ls[0],pred_dt_ls[-1]) +
              "XGB models with all features have been fitted")
    else:
        print("  This script only plot the feature source/time step importance for the time steps:\n    %s" % pred_dt_ls)
    source_dict = merge_two_dicts(cfg_op["source_dict"], cfg_op["source_dict_combi"])

    score_ls    = []
    model_ls    = []
    for i, pred_dt in enumerate(pred_dt_ls):
        model_path_xgb = os.path.join(path_xgb,"model_%i_t0diff_maxdepth6.pkl" % pred_dt)
        with open(model_path_xgb,"rb") as file:
            xgb_dt = pickle.load(file)
        score_ls.append(pd.DataFrame.from_dict(xgb_dt.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["%imin" % pred_dt]))
        model_ls.append(xgb_dt)

    feature_names = model_ls[0].get_booster().feature_names
    df_gain = pd.concat([pd.DataFrame([],index=feature_names)]+score_ls,axis=1,sort=True).fillna(0, inplace=False).astype(np.float32)

    feat_ls = list(df_gain.index)
    feat_var_ls = [feat.split("|")[0] for feat in feat_ls]
    feat_dt_ls  = [int(feat.split("|")[1]) if "|" in feat else 0 for feat in feat_ls]

    Time_feat      = [u'SOLAR_TIME_COS', u'SOLAR_TIME_SIN']
    TRT_Rank_feat  = [u'RANK', u'RANKr', u'TRT_Rank']
    TRT_light_feat = [u'CG', u'CG_minus', u'CG_plus', u'perc_CG_plus']
    TRT_radar_feat = [u'CZC_lt57dBZ',  u'ET15', u'ET15m', u'ET45', u'ET45m', u'POH', u'VIL',  u'maxH', u'maxHm']
    TRT_loc_feat   = [u'Border_cell', u'Dvel_x', u'Dvel_y', u'angle', u'area', u'det', u'ell_L', u'ell_S', u'iCH', u'jCH', u'lat', u'lon', u'vel_x', u'vel_y']

    feat_src_ls = [source_dict[feat_var.split("_stat")[0]] if "_stat" in feat_var else feat_var for feat_var in feat_var_ls]
    feat_src_ls = [source_dict[feat_var.split("_pixc")[0]] if "_pixc" in feat_var else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["TRT_Rank" if feat_var in TRT_Rank_feat  else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["RADAR"    if feat_var in TRT_radar_feat else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["THX"      if feat_var in TRT_light_feat else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["LOC_AREA" if feat_var in TRT_loc_feat   else feat_var for feat_var in feat_src_ls]
    feat_src_ls = ["TIME"     if feat_var in Time_feat      else feat_var for feat_var in feat_src_ls]

    df_gain = pd.concat([df_gain, pd.DataFrame.from_dict({"TIME_DELTA": feat_dt_ls, "VARIABLE": feat_var_ls, "SOURCE": feat_src_ls}).set_index(df_gain.index)], axis=1)
    df_gain["SOURCE"]   = df_gain["SOURCE"].astype('category')
    df_gain["VARIABLE"] = df_gain["VARIABLE"].astype('category')

    df_sum_source      = df_gain.groupby("SOURCE").sum().transpose().drop("TIME_DELTA")
    df_sum_source_norm = df_sum_source.div(df_sum_source.sum(axis=1), axis=0)
    #df_sum_source_norm = df_sum_source_norm.iloc[:,[3,4,0,5,2,7,1,6]]

    df_sum_dtime       = df_gain.groupby("TIME_DELTA").sum().transpose()
    df_sum_dtime_norm  = df_sum_dtime.div(df_sum_dtime.sum(axis=1), axis=0)
    df_sum_dtime_norm.columns = ["%imin" % colname for colname in df_sum_dtime_norm.columns]

    fig = plt.figure(figsize = [10,13])
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    cmap = plt.get_cmap('inferno')
    past_dt_cmap = truncate_colormap(cmap, 0.0, 0.8)
    df_sum_source_norm.plot.line(ax=ax1, cmap="Set1", linewidth=1.5)
    df_sum_dtime_norm.plot.line(ax=ax2, cmap=past_dt_cmap, linewidth=1.5)
    for title,ax in zip(["Feature source","Past time step"],[ax1,ax2]):
        box = ax.get_position()
        #ax.patch.set_facecolor((0.9,0.9,0.9))
        ax.set_position([box.x0, box.y0, # + box.height * 0.1,
                         box.width, box.height * 0.8])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
                  fancybox=True, shadow=True, ncol=5, title=title, fontsize="medium")
        ax.set_xlabel("Lead time")
        ax.grid()
        ax.grid(axis='x',linewidth=4,alpha=0.5)
        #ax.tick_params(axis='y', colors = "viridis_r") #, grid_color = "viridis_r")
        cmap = plt.cm.get_cmap('viridis_r')
        n_xticks = range(len(ax.get_xticklines()))
        for i, tick, label, gridl in zip(n_xticks, ax.get_xticklines(), ax.get_xticklabels(), ax.get_xgridlines()):
            col_tick = cmap(float(i)/len(ax.get_xticklines()))
            tick.set_color(col_tick)
            label.set_color(col_tick)
            gridl.set_color(col_tick)
    ax1.set_ylabel("Relative feature source importance")
    ax2.set_ylabel("Relative time step importance")
    #plt.tight_layout()
    plt_saving_location = os.path.join(cfg_tds["fig_output_path"],"Feature_source_time_step_importance.pdf")
    plt.savefig(plt_saving_location,orientation="portrait")
    print("  Plot saved in:\n    %s" % plt_saving_location)

## Plot XGB model weights (to push importance of strong TRT cells which are not decreasing):
def plot_XGB_model_weights(df_nonnan_nonzerot0t10, cfg_tds):
    weights_df = df_nonnan_nonzerot0t10[["TRT_Rank|0","TRT_Rank_diff|10"]]
    weights_df["Weight"] = np.exp(weights_df["TRT_Rank|0"]) * np.exp(weights_df["TRT_Rank|0"]+weights_df["TRT_Rank_diff|10"])
    weights_df.plot.scatter(x="TRT_Rank|0",y="TRT_Rank_diff|10",c="Weight",marker="D", s=1, cmap="plasma")

    fig, axes = plt.subplots(1,2)
    fig.set_size_inches(8,4)
    weights_df.plot.scatter(ax=axes[0],x="TRT_Rank|0",y="TRT_Rank_diff|10",c="Weight",marker="D", s=1, cmap="plasma")
    axes[0].set_title("Model Weights\n ")
    weights_df.plot.scatter(ax=axes[1],x="TRT_Rank|0",y="TRT_Rank_diff|10",c="Weight",marker="D", s=1, cmap="plasma",norm=mcolors.LogNorm())
    axes[1].set_title("Model Weights\n (logarithmic)")

    for ax in axes:
        ax.set_ylabel(r"TRT Rank change t$\mathregular{_{+10min}}$") #; axes.set_title('TRT Ranks (16km diameter)')
        ax.set_xlabel(r"TRT Rank  $\mathregular{t_0}$")
        ax.set_aspect('equal')
        ax.patch.set_facecolor('0.7')
        ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Model_weights.pdf"), orientation="portrait")


















