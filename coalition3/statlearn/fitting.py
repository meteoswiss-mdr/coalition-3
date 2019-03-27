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
import coalition3.statlearn.feature as feat
from coalition3.visualisation.TRTcells import contour_of_2dHist
from coalition3.visualisation.TRTcells import truncate_cmap

## =============================================================================
## FUNCTIONS:

## Get cv MLP model:
def cv_mlp_model(X_train, y_train, n_feat, verbose_bool=False):
    alpha_vals = [1e+0, 1e-3, 1e-6]
    hidden_layer_setup = [(int(np.ceil(n_feat*1.5)),int(np.ceil(n_feat*1.5))), # (n_feat,int(np.ceil(n_feat/2.))),
                          (int(np.ceil(n_feat*2/3.)),int(np.ceil(n_feat*1/3.)))]
    param_grid = {'alpha': alpha_vals, 'hidden_layer_sizes': hidden_layer_setup}
    gs_object = GridSearchCV(MLPRegressor(verbose=verbose_bool), param_grid=param_grid, cv=3)
    gs_object.fit(X_train.values, y_train.values)
    return gs_object

## Get Mean Square Error (MSE) for different number of n features:
def fit_model_n_feat(X_train, y_train, top_features, n_feat, n_feat_arr,
                     model="xgb", cv=True, verbose_bool=False, set_log_weight=False):
    perc_finished = np.sum(n_feat_arr[:np.where(n_feat_arr==n_feat)[0][0]],
                           dtype=np.float32) / np.sum(n_feat_arr,dtype=np.float32)
    ending = "\n" if verbose_bool else "\r"
    print("    Working on %3i features (~%4.1f%%)" % (n_feat,perc_finished*100),end=ending)
    sys.stdout.flush()
    if model=="xgb":
        if set_log_weight:
            if np.where(n_feat_arr==n_feat)[0][0] == 0:
                print("\n      Using log weights for XGB model fitting")
            s_weights = X_train["s_weight"].values
            X = X_train.drop(labels="s_weight", axis=1)
        else:
            s_weights = None
        model = xgb.XGBRegressor(max_depth=6,silent=True,n_jobs=6,nthreads=6)
        model.fit(X_train[top_features.index[:n_feat]], y_train,
                  sample_weight=s_weights)
    elif model=="mlp":
        if cv:
            model = cv_mlp_model(X_train[top_features.index[:n_feat]], y_train,
                                 n_feat, verbose_bool)
        else:
            model = MLPRegressor(hidden_layer_sizes=(n_feat,np.ceil(n_feat/2.)))
            model.fit(X_train[top_features.index[:n_feat]], y_train)
    return(model)
    #return(mean_squared_error(y_test, model.predict(X_test[top_features.index[:n_feat]])))

## Get Mean Square Error (MSE) and R^2 for different number of n features:
def mse_r2_n_feat(X_test, y_test, top_features, n_feat, model):
    if isinstance(model, sklearn.model_selection._search.GridSearchCV):
        model_pred = model.best_estimator_
    else:
        model_pred = model
    if isinstance(top_features, list):
        prediction = model_pred.predict(X_test[top_features[:n_feat]])
    else:
        prediction = model_pred.predict(X_test[top_features.index[:n_feat]])
    mse_val    = sklearn.metrics.mean_squared_error(y_test,prediction)
    r2_val     = sklearn.metrics.r2_score(y_test,prediction)
    return(mse_val, r2_val)


def get_n_feat_arr(model):
    n_feat_arr = np.concatenate([np.arange(10)+1,
                                 np.arange(12,40,2),
                                 np.arange(40,100,20),
                                 np.arange(100,200,25),
                                 np.arange(200,300,50),
                                 np.arange(300,500,100),
                                 np.arange(500,1250,250)])
    if model == "xgb":
        return n_feat_arr
    elif model in ["ann","mlp"]:
        return n_feat_arr[9:]

## Get Mean Square Error depending on number of features:
def get_mse_from_n_feat(df_nonnan_nonzerot0,pred_dt,cfg_tds,model_path,
                        mod_bound=None,mod_name="",delete_RADAR_t0=False,
                        set_log_weight=False):
    print("Get dependence of MSE on n features for lead time t0 + %imin" % pred_dt, end="")
    if mod_bound is not None:
        print(" (for %s)" % mod_name)
        mod_name = "_%s" % mod_name
    else:
        print(" (for all samples)")
    sys.stdout.flush()

    ## Check whether data on MSE already exists:
    calc_new_model = "y"
    if os.path.exists(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name))):
        calc_new_model = ""
        while (calc_new_model!="y" and calc_new_model!="n"):
            calc_new_model = raw_input("  MSE data exists alreay, get new one? [y/n] ")
        #if calc_new_model=="n":
        #    print("  Use existing one, return from this function")
        #    return

    ## Calculate sample weights for XGB fitting:
    if set_log_weight:
        df_nonnan_nonzerot0["s_weight"] = feat.calc_sample_weight(df_nonnan_nonzerot0["TRT_Rank|0"],
                                                                  df_nonnan_nonzerot0["TRT_Rank_diff|%i" % pred_dt])

    ## Delete rows with TRT Rank close to zero at lead time:
    print("  Delete rows with TRT Rank close to zero at lead time")
    if delete_RADAR_t0:
        print("  Get predictor matrix X without RADAR variables at t0")
        X_feature_sel = "no_radar_t0"
    else:
        print("  Get predictor matrix X with RADAR variables at t0")
        X_feature_sel = "all"
    X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
            del_TRTeqZero_tpred=True, split_Xy_traintest=True,
            pred_dt=pred_dt, TRTRankt0_bound=mod_bound,check_for_nans=False,
            X_feature_sel=X_feature_sel)

    ## Load XGBmodel:
    print("  Load XGBmodel")
    with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"rb") as file:
        xgb_model = pickle.load(file)

    ## Order features by importance (gain):
    top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                               orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                               ascending=False)

    ## Create list of number of features to select for the fitting:
    n_feat_arr = get_n_feat_arr(model="xgb")

    ## Get models fitted with n top features:
    if calc_new_model=="y":
        print("  Get models fitted with n top features")
        ls_models = [fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, n_feat_arr, set_log_weight=set_log_weight) for n_feat in n_feat_arr]
        print("    Save list of models as pickle to disk")
        with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"wb") as file:
            pickle.dump(ls_models, file, protocol=2)
    else:
        print("  Load existing models fitted with n top features")
        with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"rb") as file:
            ls_models = pickle.load(ls_models, file, protocol=2)

    ## Get mean square error of models with n features:
    print("  Get mean square error of models with n features")
    MSE_r2_ls = [mse_r2_n_feat(X_test, y_test, top_features_gain, n_feat, model) \
                   for n_feat, model in zip(n_feat_arr,ls_models)]
    df_mse_feat_count = pd.DataFrame.from_dict({"Feature Count": n_feat_arr,
        "MSE %imin%s" % (pred_dt,mod_name): [score[0] for score in MSE_r2_ls],
         "R2 %imin%s" % (pred_dt,mod_name): [score[1] for score in MSE_r2_ls]})
    df_mse_feat_count.set_index("Feature Count",inplace=True)
    print("    Save dataframe with MSE to disk")
    with open(os.path.join(model_path,"MSE_feature_count_gain_%i%s.pkl" % (pred_dt,mod_name)),"wb") as file:
        pickle.dump(df_mse_feat_count,file,protocol=2)

    ## Append MSE values to existing HDF5 file (if existing):
    print("  Append MSE values to HDF5 file")
    df_mse_feat_count.to_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                             key="MSE_%imin%s" % (pred_dt,mod_name), mode="a", format="t", append=True)

## Plot dependence of MSE on feature number:
def plot_mse_from_n_feat(ls_pred_dt,cfg_tds,model_path,thresholds=None,ls_model_names=[""]):
    print("Plot MSE as function of number of features used")
    if ls_model_names != [""]:
        ls_model_names = ["_%s" % name for name in ls_model_names]

    ## Read in MSE values:
    print("  Reading MSE values from disk")
    ls_df_mse_feat_count = []
    for model_name in ls_model_names:
        ls_df_mse_feat_count += [pd.read_hdf(os.path.join(model_path,"MSE_feature_count_gain.h5"),
                                             key="MSE_%imin%s" % (pred_dt,model_name), mode="r") for pred_dt in ls_pred_dt]
    df_mse_feat_count = pd.concat(ls_df_mse_feat_count, axis=1)

    ## Plot the figure:
    print("  Plotting the figure")
    col10 = '#E69F00'
    col30 = '#D55E00'

    if (len(ls_pred_dt)>0 and ls_model_names == [""]):
        #df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse = df_mse_feat_count[[MSE_col for MSE_col in df_mse_feat_count.columns if "MSE" in MSE_col]]
        df_r2  = df_mse_feat_count[[R2_col  for R2_col  in df_mse_feat_count.columns if "R2"  in R2_col]]
        df_mse.columns = [colname[3:] for colname in df_mse.columns]
        df_r2.columns  = [colname[3:] for colname in df_r2.columns]
        fig, axes = plt.subplots(1, 2, figsize = (8,6))
        df_mse.plot(ax = axes[0], linestyle="-", cmap="viridis_r",legend=False)
        axes[0].set_ylabel(r'Mean square error MSE')
        df_r2.plot(ax = axes[1], linestyle="-", cmap="viridis_r")
        axes[1].set_ylabel(r'Coeff of determination $\mathregular{R^2}$')
        plt.tight_layout()
        for ax in axes:
            ax.grid()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],
                                         "MSE_R2_feature_count.pdf"),
                    orientation="portrait")
    elif (len(ls_pred_dt) == 3 and ls_model_names == [""]):
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm
        df_mse_feat_count_norm.plot(ax = ax, linestyle="-", cmap="Set1")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    elif (len(ls_pred_dt) == 2 and ls_model_names == [""]):
        fig = plt.figure(figsize = [10,5])
        ax1 = fig.add_subplot(1,1,1)
        ax2 = ax1.twinx()
        ax1.plot(df_mse_feat_count.index,df_mse_feat_count[["MSE %imin" % ls_pred_dt[0]]], col10, label="%imin" % ls_pred_dt[0])
        ax2.plot(df_mse_feat_count.index,df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]], col30, label="%imin" % ls_pred_dt[1])
        #ax1.semilogx(n_feat_arr,df_mse_feat_count[["MSE 10min"]], '#E69F00', label='10min')
        #ax2.semilogx(n_feat_arr,df_mse_feat_count[["MSE 30min"]], '#56B4E9', label='30min')
        #df_mse_feat_count.plot(ax=ax1,x="Feature Count",y="MSE 10min", style='b-', secondary_y=False)
        #df_mse_feat_count.plot(ax=ax2,x="Feature Count",y="MSE 30min", style='g-', secondary_y=True)
        ax1.set_title("Mean square error (MSE) as function of feature count")
        ax1.set_xlabel(r"Number of features")
        ax1.set_ylabel("MSE - %imin prediction" % ls_pred_dt[0], color=col10)
        ax2.set_ylabel("MSE - %imin prediction" % ls_pred_dt[1], color=col30)
        ax1.grid()
        #ax2.legend([ax1.get_lines()[0], ax2.right_ax.get_lines()[0]], ['A','B'], bbox_to_anchor=(1.5, 0.5))
        #plt.show()
        if thresholds is not None:
            props = dict(boxstyle='round', facecolor='white')
            ax2.axvline(x=thresholds[0], color=col10, linestyle='--', linewidth=2)
            ax2.axvline(x=thresholds[1], color=col30, linestyle='--', linewidth=2)
            ax2.text(thresholds[0], df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]].values[3],
                     "Treshold %imin" % ls_pred_dt[0], verticalalignment='top', bbox=props)
            ax2.text(thresholds[1], df_mse_feat_count[["MSE %imin" % ls_pred_dt[1]]].values[3],
                     "Treshold %imin" % ls_pred_dt[1], verticalalignment='top', bbox=props)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
        else:
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_%i.pdf"), orientation="portrait")
            plt.close()
    elif (len(ls_pred_dt) == 2 and ls_model_names != [""]):
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm.columns = [colname.replace("_", " (")+")" for colname in df_mse_feat_count_norm.columns]
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        if len(ls_model_names)==2:
            df_mse_feat_count_norm.iloc[:,:2].plot(ax = ax, color=[col10,col30])
            df_mse_feat_count_norm.iloc[:,2:].plot(ax = ax, linestyle="--", color=[col10,col30])
        else:
            ax = df_mse_feat_count_norm.iloc[:,::2].plot(cmap="Paired")
            df_mse_feat_count_norm.iloc[:,1::2].plot(ax = ax, linestyle="--", cmap="Paired")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        if thresholds is not None and len(ls_model_names)==2:
            ps = dict(boxstyle='round', facecolor='white')
            ax.axvline(x=thresholds[0], color=col10, linestyle='solid', linewidth=2)
            ax.axvline(x=thresholds[1], color=col30, linestyle='solid', linewidth=2)
            ax.axvline(x=thresholds[2], color=col10, linestyle='--', linewidth=2)
            ax.axvline(x=thresholds[3], color=col30, linestyle='--', linewidth=2)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
            plt.close()
        else:
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    elif (len(ls_pred_dt) == 1 and ls_model_names != [""]):
        df_mse_feat_count_norm = df_mse_feat_count/df_mse_feat_count.mean()
        df_mse_feat_count_norm.columns = [colname.replace("_", " (")+")" for colname in df_mse_feat_count_norm.columns]
        fig = plt.figure(figsize = [10,5])
        ax = fig.add_subplot(1,1,1)
        if len(ls_model_names)==1:
            df_mse_feat_count_norm.iloc[:,:2].plot(ax = ax, color=[col10,col30])
            #df_mse_feat_count_norm.iloc[:,2:].plot(ax = ax, linestyle="--", color=[col10,col30])
        else:
            ax = df_mse_feat_count_norm.iloc[:,::2].plot(cmap="Paired")
            df_mse_feat_count_norm.iloc[:,1::2].plot(ax = ax, linestyle="--", cmap="Paired")
        ax.set_ylabel("Normalised MSE")
        ax.set_xlabel("Number of features")
        ax.set_title("Normalised mean square error (MSE) as function of feature count")
        ax.grid()
        if thresholds is not None and len(ls_model_names)==2:
            ps = dict(boxstyle='round', facecolor='white')
            ax.axvline(x=thresholds[0], color=col10, linestyle='solid', linewidth=2)
            #ax.axvline(x=thresholds[1], color=col30, linestyle='solid', linewidth=2)
            #ax.axvline(x=thresholds[2], color=col10, linestyle='--', linewidth=2)
            #ax.axvline(x=thresholds[3], color=col30, linestyle='--', linewidth=2)
            plt.pause(8)
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count_thresh.pdf"), orientation="portrait")
            plt.close()
        else:
            plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")
    else:
        df_mse_feat_count.plot.line()
        plt.savefig(os.path.join(cfg_tds["fig_output_path"],"MSE_feature_count.pdf"), orientation="portrait")

## Fit model with threshold number of features (selected by the user):
def plot_pred_vs_obs(df_nonnan_nonzerot0,pred_dt,n_feat_ls,cfg_tds,model_path,
                     ls_mod_bound=[None],ls_model_names=[""]):
    if len(ls_mod_bound)>1:
        y_test_ls = []
        pred_gain_ls = []

    for mod_bound, n_feat, mod_name in zip(ls_mod_bound,n_feat_ls,ls_model_names):
        print("\nPlot predicted vs. observed testing samples for time delta %imin" % (pred_dt),end="")
        if mod_bound is not None:
            print(" (%i features for model '%s')" % (n_feat,mod_name))
            mod_name = "_%s" % mod_name
        else:
            print(" (%i features for all samples)" % n_feat)
        sys.stdout.flush()

        ## Delete rows with TRT Rank close to zero at lead time:
        print("  Delete rows with TRT Rank close to zero at lead time")
        X_train, X_test, y_train, y_test = ipt.get_model_input(df_nonnan_nonzerot0,
                del_TRTeqZero_tpred=True, split_Xy_traintest=True,
                pred_dt=pred_dt, TRTRankt0_bound=mod_bound,check_for_nans=False,
                )

        precalc_n_feat = get_n_feat_arr("xgb")

        if n_feat not in precalc_n_feat:
            ## Load XGBmodel:
            print("  Load XGBmodel")
            with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6.pkl" % (pred_dt,mod_name)),"rb") as file:
                xgb_model = pickle.load(file)

            ## Order features by importance (gain):
            top_features_gain = pd.DataFrame.from_dict(xgb_model.get_booster().get_score(importance_type='gain'),
                                                       orient="index",columns=["F_score"]).sort_values(by=['F_score'],
                                                       ascending=False)

            ## Fit model:
            model    = fit_model_n_feat(X_train, y_train, top_features_gain, n_feat, np.array(n_feat))
        else:
            with open(os.path.join(model_path,"models_%i%s_t0diff_maxdepth6_nfeat.pkl" % (pred_dt,mod_name)),"rb") as file:
                model = pickle.load(file)[np.where(precalc_n_feat==n_feat)[0][0]]

        ## Get features:
        features  = model.get_booster().feature_names

        ## Make prediction and get skill scores:
        pred_gain = model.predict(X_test[features])
        mse_gain  = sklearn.metrics.mean_squared_error(y_test, pred_gain)
        r2_gain   = sklearn.metrics.r2_score(y_test, pred_gain)

        ## Save the model to disk:
        with open(os.path.join(model_path,"model_%i%s_t0diff_maxdepth6_%ifeat_gain.pkl" % (pred_dt,mod_name,n_feat)),"wb") as file:
            pickle.dump(model,file,protocol=2)

        ## Make the plot:
        plot_pred_vs_obs_core(y_test,pred_gain,pred_dt,
                              mod_name,cfg_tds,outtype="TRT_Rank_diff")

        plot_pred_vs_obs_core(X_test["TRT_Rank|0"]+y_test,
                              X_test["TRT_Rank|0"]+pred_gain,pred_dt,
                              mod_name,cfg_tds,outtype="TRT_Rank")

        ## Append to list of results for combined plot:
        if len(ls_mod_bound)>1:
            y_test_ls.append(y_test)
            pred_gain_ls.append(pred_gain)

    ## Make combined plot:
    if len(ls_mod_bound)>1:
        y_test_combi = pd.concat(y_test_ls,axis=0)
        pred_gain_combi = np.concatenate(pred_gain_ls)
        mse_gain  = sklearn.metrics.mean_squared_error(y_test_combi,pred_gain_combi)
        r2_gain   = sklearn.metrics.r2_score(y_test_combi,pred_gain_combi)
        plot_pred_vs_obs_core(y_test_combi, pred_gain_combi,
                              pred_dt,"_%s" % "|".join(ls_model_names),cfg_tds)


def plot_pred_vs_obs_core(y_test,pred_gain,pred_dt,mod_name,cfg_tds,outtype="TRT_Rank_diff"):
    print("  Making the plot")
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[10,8])
    if outtype=="TRT_Rank_diff":
        fig_range = [-2.5,2.5]
        print_str = "difference "
        save_str = "_diff"
    elif outtype=="TRT_Rank":
        fig_range = [0,4]
        print_str = ""
        save_str = "_rank"

    if len(y_test)>1000:
        counts,ybins,xbins,image = axes.hist2d(y_test.values,pred_gain,
                                               bins=200,range=[fig_range,fig_range],cmap="magma",norm=mcolors.LogNorm())
        #counts,ybins,xbins,image = axes.hist2d(y_test[["TRT_Rank_diff|%i" % pred_dt]].values[:,0],pred_gain,
        #                                       bins=200,range=[[-2.5,2.5],[-2.5,2.5]],cmap="magma",norm=mcolors.LogNorm())
        cbar = fig.colorbar(image, ax=axes, extend='max')
    else:
        axes.scatter(y_test[["%s|%i" % (outtype,pred_dt)]].values[:,0],pred_gain,
                     marker="+", color="black", s=8)
    axes.set_xlim(fig_range); axes.set_ylim(fig_range)
    #cbar.set_label('Number of cells per bin of size [0.02, 0.02]', rotation=90)
    axes.grid()
    #axes.fill_between([-0.2,0.2],y1=[-1.5,-1.5], y2=[1.5,1.5], facecolor="none", hatch="X", edgecolor="darkred", linewidth=0.5)
    axes.plot(fig_range,fig_range,'w--',linewidth=2) #,facecolor="w",linewidth=2,linestyle='--')
    if len(y_test)>1000:
        cont2d_1, lvl = contour_of_2dHist(counts,smooth=True)
        CS = axes.contour(cont2d_1,levels=lvl,extent=[xbins.min(),xbins.max(),ybins.min(),ybins.max()],linewidths=2,cmap="YlGn_r")
        CS_lab = axes.clabel(CS, inline=1, fontsize=10, fmt='%i%%', colors="black")
        #[txt.set_backgroundcolor('white') for txt in CS_lab]
        [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.3, boxstyle='round', alpha=0.71)) for txt in CS_lab] #pad=0,
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test.values,pred_gain)
    axes.plot(fig_range,np.array(fig_range)*slope+intercept,'darkred',linewidth=2)
    
    axes.set_xlabel(r'Observed TRT Rank %st$\mathregular{_{+%imin}}$' % (print_str,pred_dt))
    axes.set_ylabel(r'Predicted TRT Rank %st$\mathregular{_{+%imin}}$' % (print_str,pred_dt))
    model_title = "" if mod_name == "" else r" | Mod$\mathregular{_{%s}}$" % mod_name[1:]
    title_str = 'TRT Ranks %s\nTime delta: %imin' % (print_str,pred_dt)
    title_str += model_title
    axes.set_title(title_str)
    axes.set_aspect('equal'); axes.patch.set_facecolor('0.71')
    
    mse_gain = sklearn.metrics.mean_squared_error(y_test.values,pred_gain)
    r2_gain  = sklearn.metrics.r2_score(y_test.values,pred_gain)
    str_n_cells  = "Mean Squared Error (MSE): %.2f\n" % (mse_gain)
    str_n_cells += r"Coeff of determination (R$\mathregular{^2}$): %.2f" % (r2_gain); str_n_cells += "\n"
    str_n_cells += r"Regression intercept ($\mathregular{\beta_0}$): %.2f" % (intercept); str_n_cells += "\n"
    str_n_cells += r"Regression slope ($\mathregular{\beta_1}$): %.2f" % (slope)
    props = dict(boxstyle='round', facecolor='white')
    axes.text(fig_range[0]+0.25, fig_range[1]-0.25, str_n_cells, bbox=props,
              horizontalalignment='left',verticalalignment='top')
    plt.savefig(os.path.join(cfg_tds["fig_output_path"],"Pred_scatter%s_%i%s.pdf" % (save_str,pred_dt,mod_name.replace("|","-"))), orientation="portrait")
    print("    Saved the plot")
    plt.close()

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


















